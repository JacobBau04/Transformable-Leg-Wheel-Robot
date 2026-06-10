"""Export a trained RMA policy to a framework-free .npz for on-robot deployment.

The deploy target (Jetson/Pi) has no JAX. The networks are tiny, so we dump just
the weights the robot needs into a flat .npz and reimplement the forward passes
in pure numpy (see packages/twmr/src/twmr/deploy.py).

What the robot needs at deployment (and ONLY this):
  * observation normalizer mean/std for the 29-dim student slice. The privileged
    slice (29:38) and the EnvFactorEncoder mu are NOT deployed -- phi replaces mu.
  * pi MLP weights (the `apply_with_z` path): Dense_0..Dense_3.
  * phi (AdaptationModule) weights: Dense_0..Dense_5.

Usage:
    python export_params.py                       # auto-discover latest matched pair
    python export_params.py --run-dir logs/TWMRLegTerr-20260528-160033
    python export_params.py --ppo PATH --phi PATH --out deploy_params.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from brax.io import model

from twmr.networks import STUDENT_OBS_SIZE, ENV_LATENT_SIZE
from twmr.adaptation import HISTORY_LEN, PER_STEP_FEAT, ACTION_SIZE, AdaptationModule

# π MLP (TeacherPolicyNetwork._mlp) Dense layer names, in forward order. The
# encoder lives under EnvFactorEncoder_0 and is intentionally excluded.
_PI_LAYERS = ["Dense_0", "Dense_1", "Dense_2", "Dense_3"]
# φ (AdaptationModule) Dense layer names: 2 per-step + 3 windowed-conv + 1 final.
_PHI_LAYERS = ["Dense_0", "Dense_1", "Dense_2", "Dense_3", "Dense_4", "Dense_5"]


def _discover_pair(env_name: str) -> tuple[Path, Path]:
    """Latest run dir that has BOTH ppo_final and phi_final (a deployable pair)."""
    runs = sorted(
        Path("logs").glob(f"{env_name}-[0-9]*/checkpoints"),
        key=lambda p: p.stat().st_mtime,
    )
    for ck in reversed(runs):
        if (ck / "ppo_final").exists() and (ck / "phi_final").exists():
            return ck / "ppo_final", ck / "phi_final"
    raise FileNotFoundError(
        f"No run under logs/ has both ppo_final and phi_final for {env_name!r}. "
        f"Run phase1_run.ipynb then phase2_run.ipynb first."
    )


def _layer(params_tree: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    sub = params_tree[name]
    return np.asarray(sub["kernel"], np.float32), np.asarray(sub["bias"], np.float32)


def export(ppo_path: Path, phi_path: Path, out_path: Path) -> dict:
    norm, policy, _value = model.load_params(str(ppo_path))
    phi = model.load_params(str(phi_path))

    pol = policy["params"]
    phi_p = phi["params"]

    # Fail loudly if the checkpoint layout drifts from what we expect.
    assert "EnvFactorEncoder_0" in pol, "expected mu encoder in policy params"
    missing = [k for k in _PI_LAYERS if k not in pol]
    assert not missing, f"pi MLP layers missing from checkpoint: {missing}"
    missing = [k for k in _PHI_LAYERS if k not in phi_p]
    assert not missing, f"phi layers missing from checkpoint: {missing}"

    mean = np.asarray(norm.mean, np.float32)
    std = np.asarray(norm.std, np.float32)
    assert mean.shape[-1] >= STUDENT_OBS_SIZE, f"norm mean too short: {mean.shape}"

    out: dict[str, np.ndarray] = {}
    # Only the student slice is normalized + fed to pi/phi on-robot.
    out["norm_mean"] = mean[:STUDENT_OBS_SIZE]
    out["norm_std"] = std[:STUDENT_OBS_SIZE]

    for i, name in enumerate(_PI_LAYERS):
        w, b = _layer(pol, name)
        out[f"pi_W{i}"] = w
        out[f"pi_b{i}"] = b

    for i, name in enumerate(_PHI_LAYERS):
        w, b = _layer(phi_p, name)
        out[f"phi_W{i}"] = w
        out[f"phi_b{i}"] = b

    # Architecture metadata so deploy.py stays self-contained (numpy-only, no twmr import).
    m = AdaptationModule()
    out["meta_student_obs_size"] = np.int64(STUDENT_OBS_SIZE)
    out["meta_latent_dim"] = np.int64(ENV_LATENT_SIZE)
    out["meta_action_size"] = np.int64(ACTION_SIZE)
    out["meta_history_len"] = np.int64(HISTORY_LEN)
    out["meta_per_step_feat"] = np.int64(PER_STEP_FEAT)
    out["meta_phi_n_per_step"] = np.int64(len(m.per_step_hidden))   # per-step Dense layers
    out["meta_phi_conv_specs"] = np.asarray(m.conv_specs, np.int64)  # (n_conv, 2): kernel, stride
    out["meta_n_pi_layers"] = np.int64(len(_PI_LAYERS))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    return out


def _verify_roundtrip(out_path: Path, exported: dict) -> None:
    """Step 2 test gate: reload and confirm everything round-trips, mu/value absent."""
    z = np.load(out_path)
    keys = set(z.files)
    for k, v in exported.items():
        assert k in keys, f"key {k} missing after reload"
        np.testing.assert_array_equal(z[k], v)
    # No mu (EnvFactorEncoder) or value-net weights should have leaked in.
    assert not any("EnvFactorEncoder" in k or "value" in k.lower() for k in keys)
    assert z["norm_mean"].shape == (STUDENT_OBS_SIZE,)
    assert z["pi_W0"].shape == (PER_STEP_FEAT, 64)        # 37 = student(29)+z(8)
    assert z["phi_W5"].shape[1] == ENV_LATENT_SIZE        # final -> 8
    print("  round-trip OK; mu/value absent; shapes correct.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env-name", default="TWMRLegTerr")
    ap.add_argument("--run-dir", type=Path, help="logs/<run>/ (uses its checkpoints/)")
    ap.add_argument("--ppo", type=Path, help="explicit ppo_final path")
    ap.add_argument("--phi", type=Path, help="explicit phi_final path")
    ap.add_argument("--out", type=Path, default=Path("deploy_params.npz"))
    args = ap.parse_args()

    if args.ppo and args.phi:
        ppo_path, phi_path = args.ppo, args.phi
    elif args.run_dir:
        ppo_path = args.run_dir / "checkpoints" / "ppo_final"
        phi_path = args.run_dir / "checkpoints" / "phi_final"
    else:
        ppo_path, phi_path = _discover_pair(args.env_name)

    print(f"ppo_final : {ppo_path}")
    print(f"phi_final : {phi_path}")
    exported = export(ppo_path, phi_path, args.out)
    print(f"wrote     : {args.out}  ({len(exported)} arrays)")
    _verify_roundtrip(args.out, exported)


if __name__ == "__main__":
    main()

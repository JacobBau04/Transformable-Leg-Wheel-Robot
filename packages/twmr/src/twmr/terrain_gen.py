#!/usr/bin/env python3
"""
generate_bumpy_hfield.py

Generates a MuJoCo heightfield as a bumpy terrain, with
controllable spatial frequency statistics (mean & std, in cycles/meter).

Outputs MuJoCo binary heightfield format:
  (int32) nrow
  (int32) ncol
  (float32) data[nrow*ncol]   row-major
MuJoCo normalizes loaded elevation data to [0,1]


"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np


def bandlimited_noise_heightfield(
    nrow: int,
    ncol: int,
    side_m: float,
    freq_mean_cpm: float,
    freq_std_cpm: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Create a 2D heightfield by filtering white noise in the Fourier domain with a
    radial Gaussian band-pass centered at freq_mean_cpm (cycles/m).

    Returns float32 array shape (nrow, ncol) with arbitrary range; caller may normalize.
    """
    if nrow <= 2 or ncol <= 2:
        raise ValueError("nrow and ncol must be > 2.")
    if side_m <= 0:
        raise ValueError("side_m must be > 0.")
    if freq_mean_cpm < 0:
        raise ValueError("freq_mean_cpm must be >= 0.")
    if freq_std_cpm <= 0:
        raise ValueError("freq_std_cpm must be > 0 (try 0.05–0.5).")

    rng = np.random.default_rng(seed)

    # White noise in spatial domain (real), so FFT has Hermitian symmetry.
    noise = rng.standard_normal((nrow, ncol), dtype=np.float64)

    # Frequency axes in cycles/meter (not radians/m).
    # Using periodic length = side_m for both dimensions.
    fy = np.fft.fftfreq(nrow, d=side_m / nrow)  # cycles/m
    fx = np.fft.fftfreq(ncol, d=side_m / ncol)  # cycles/m
    FX, FY = np.meshgrid(fx, fy)
    FR = np.sqrt(FX * FX + FY * FY)  # radial frequency in cycles/m

    # Radial Gaussian centered at freq_mean with std = freq_std (cycles/m).
    H = np.exp(-0.5 * ((FR - freq_mean_cpm) / freq_std_cpm) ** 2)

    # Remove DC so terrain isn't dominated by a global offset.
    H[0, 0] = 0.0

    # Filter in Fourier domain.
    F = np.fft.fft2(noise)
    Ff = F * H
    h = np.fft.ifft2(Ff).real

    return h.astype(np.float32)


def normalize_01(h: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize array to [0,1] (float32)."""
    hmin = float(np.min(h))
    hmax = float(np.max(h))
    rng = hmax - hmin
    if rng < eps:
        return np.zeros_like(h, dtype=np.float32)
    out = (h - hmin) / rng
    return out.astype(np.float32)


def save_mujoco_hfield_bin(path: Path, data01: np.ndarray) -> None:
    """
    Save heightfield in MuJoCo custom binary format:
      int32 nrow, int32 ncol, float32 data row-major
    MuJoCo expects this for non-.png hfield files. :contentReference[oaicite:3]{index=3}
    """
    if data01.ndim != 2:
        raise ValueError("data01 must be a 2D array.")
    nrow, ncol = data01.shape

    # Ensure row-major float32 contiguous
    data01 = np.ascontiguousarray(data01.astype(np.float32))

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.array([nrow], dtype="<i4").tofile(f)
        np.array([ncol], dtype="<i4").tofile(f)
        data01.astype("<f4").tofile(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="terrain_height.bin",
                    help="Output MuJoCo hfield binary file (default: terrain_height.bin)")
    ap.add_argument("--side_m", type=float, default=5.0,
                    help="Physical side length in meters (default: 30.0 for 30m x 30m)")
    ap.add_argument("--nrow", type=int, default=513,
                    help="Heightfield rows (default: 513)")
    ap.add_argument("--ncol", type=int, default=513,
                    help="Heightfield cols (default: 513)")

    # Spatial frequency parameters (cycles per meter).
    ap.add_argument("--freq_mean", type=float, default=2,
                    help="Mean bump spatial frequency in cycles/m (default: 0.4 -> ~2.5m wavelength)")
    ap.add_argument("--freq_std", type=float, default=0.8,
                    help="Std dev of spatial frequency in cycles/m (default: 0.15)")

    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed (default: 0). Use different seed for different terrains.")
    ap.add_argument("--power", type=float, default=1.0,
                    help="Optional nonlinearity: raise normalized heights to this power. "
                         "1.0 = none; >1 flattens lows; <1 emphasizes lows.")

    args = ap.parse_args()

    # Generate band-limited bumpy field
    h = bandlimited_noise_heightfield(
        nrow=args.nrow,
        ncol=args.ncol,
        side_m=args.side_m,
        freq_mean_cpm=args.freq_mean,
        freq_std_cpm=args.freq_std,
        seed=args.seed,
    )

    # Normalize to [0,1] (MuJoCo will also normalize on load, but doing it here
    # makes the file predictable and lets you reshape distribution with --power). :contentReference[oaicite:4]{index=4}
    h01 = normalize_01(h)

    if args.power != 1.0:
        if args.power <= 0:
            raise ValueError("--power must be > 0.")
        h01 = np.clip(h01, 0.0, 1.0) ** float(args.power)
        h01 = normalize_01(h01)

    out_path = Path(args.out)
    save_mujoco_hfield_bin(out_path, h01)

    # Print the XML line you likely want (30m x 30m => radii 15, 15). :contentReference[oaicite:5]{index=5}
    print("\nWrote:", out_path.resolve())
    print("\nIn your XML, set the hfield size for 30m x 30m like:")
    print('  <hfield name="terrain_hf" file="terrain_height.bin" size="15 15 ELEV_Z BASE_Z"/>')
    print("\nNotes:")
    print("  - ELEV_Z is the max terrain height in meters (physical bump height).")
    print("  - BASE_Z adds thickness below the minimum height (try 0.2–1.0).")
    print("  - Typical bump wavelength ≈ 1 / freq_mean. With freq_mean=0.4, wavelength ≈ 2.5 m.")


if __name__ == "__main__":
    main()

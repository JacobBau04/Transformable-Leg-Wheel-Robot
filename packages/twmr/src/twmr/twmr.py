from pathlib import Path

from jax import Array as JaxArray
from ml_collections import config_dict
from mujoco import MjModel, mjx  # type: ignore
from mujoco.mjx import Model as MjxModel
from mujoco_playground import MjxEnv, State, dm_control_suite
from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import common

ConfigOverridesDict = dict[str, str | int | list]
_XML_PATH = Path(__file__).parent / "assets" / "wheeled_mobile_robot.xml"


def default_vision_config() -> config_dict.ConfigDict:
    return config_dict.create(
        gpu_id=0,
        render_batch_size=512,
        render_width=64,
        render_height=64,
        enable_geom_groups=[0, 1, 2],
        use_rasterizer=False,
        history=3,
    )


# TODO: check all of these default values
def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.01,
        sim_dt=0.01,
        episode_length=1000,
        action_repeat=1,
        vision=False,
        vision_config=default_vision_config(),
        impl="warp",  # TODO: cartpole uses jax
        nconmax=0,
        njmax=2,
    )


class TransformableWheelMobileRobot(MjxEnv):
    def __init__(
        self,
        # Task specific config
        config: config_dict.ConfigDict = default_config(),
        config_overrides: ConfigOverridesDict | None = None,
    ):
        super().__init__(config, config_overrides)

        self._xml_path = _XML_PATH.as_posix()
        model_xml = _XML_PATH.read_text()
        self._model_assets = common.get_assets()
        self._mj_model: MjModel = MjModel.from_xml_string(model_xml, self._model_assets)
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)  # type: ignore
        self._mj_model.opt.timestep = self.sim_dt

        # TODO: figure out vision with the madrona batch renderer

    def reset(self, rng: JaxArray) -> State:
        data = mjx_env.make_data()
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: JaxArray) -> State: ...

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def mj_model(self) -> MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> MjxModel:
        return self._mjx_model


dm_control_suite.register_environment(
    env_name="TransformableWheelMobileRobot",
    env_class=TransformableWheelMobileRobot,
    cfg_class=default_config,
)

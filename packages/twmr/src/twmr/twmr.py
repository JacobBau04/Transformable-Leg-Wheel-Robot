from jax import Array as JaxArray
from ml_collections.config_dict import ConfigDict
from mujoco import MjModel
from mujoco.mjx import Model as MjxModel
from mujoco_playground import MjxEnv, State, dm_control_suite

# def default_vision_config() -> ConfigDict: ...


def default_config() -> ConfigDict: ...


class TransformableWheelMobileRobot(MjxEnv):
    def __init__(
        self,
        # Task specific config
        config: ConfigDict = default_config(),
        config_overrides: dict[str, str | int | list] | None = None,
    ):
        super().__init__(config, config_overrides)

    def reset(self, rng: JaxArray) -> State: ...

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

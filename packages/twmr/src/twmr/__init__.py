# TODO: figure out if this is an idiomatic method for importing the packages with MuJoCo
# from .twmr import TransformableWheelMobileRobot
from .twmr import TWMRLegFlat, TWMRLegTerr, domain_randomize_model
from .twmr_torques import TWMRLegFlatRawTorques
from .networks import (
    STUDENT_OBS_SIZE,
    PRIV_OBS_SIZE,
    TEACHER_OBS_SIZE,
    ENV_LATENT_SIZE,
    EnvFactorEncoder,
    TeacherPolicyNetwork,
    TeacherValueNetwork,
    TeacherPPONetworks,
    make_teacher_ppo_networks,
)
from .adaptation import (
    HISTORY_LEN,
    ACTION_SIZE,
    PER_STEP_FEAT,
    AdaptationModule,
    make_adaptation_module,
)


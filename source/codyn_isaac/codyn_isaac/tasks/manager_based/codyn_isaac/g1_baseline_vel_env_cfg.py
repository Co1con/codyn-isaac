"""G1 velocity tracking with the Unitree foot-contact gait reward."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .g1_codyn_baseline_vel_env_cfg import (
    G1CoDynBaselineEnvCfg,
    G1CoDynBaselineEventsCfg,
    G1CoDynBaselineObservationsCfg,
    G1CoDynBaselineRewardsCfg,
)

GAIT_PERIOD = 0.8


@configclass
class G1BaselineObservationsCfg(G1CoDynBaselineObservationsCfg):
    @configclass
    class PolicyCfg(G1CoDynBaselineObservationsCfg.PolicyCfg):
        oscillator_position = None
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": GAIT_PERIOD})

    @configclass
    class CriticCfg(G1CoDynBaselineObservationsCfg.CriticCfg):
        oscillator_position = None
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": GAIT_PERIOD})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1BaselineRewardsCfg(G1CoDynBaselineRewardsCfg):
    oscillator_tracking = None
    contralateral_shoulder_coordination = None

    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": GAIT_PERIOD,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            ),
        },
    )


@configclass
class G1BaselineEventsCfg(G1CoDynBaselineEventsCfg):
    reset_oscillator = None


@configclass
class G1BaselineEnvCfg(G1CoDynBaselineEnvCfg):
    observations: G1BaselineObservationsCfg = G1BaselineObservationsCfg()
    rewards: G1BaselineRewardsCfg = G1BaselineRewardsCfg()
    events: G1BaselineEventsCfg = G1BaselineEventsCfg()


@configclass
class G1BaselinePlayEnvCfg(G1BaselineEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges.copy()
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 10

"""Reference-plus-residual targets with a weak quadratic gait deviation penalty."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import mdp
from .g1_codyn_baseline_vel_env_cfg import G1CoDynBaselineEnvCfg, G1CoDynBaselinePlayEnvCfg, G1CoDynBaselineRewardsCfg


@configclass
class G1CoDynResidualActionsCfg:
    JointPositionAction = mdp.OscillatorResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            "left_hip_pitch_joint": 0.1,
            "left_knee_joint": 0.1,
            "right_hip_pitch_joint": 0.1,
            "right_knee_joint": 0.1,
            "(?!(?:left_hip_pitch_joint|left_knee_joint|right_hip_pitch_joint|right_knee_joint)$).*": 0.25,
        },
        use_default_offset=True,
    )


@configclass
class G1CoDynResidualRewardsCfg(G1CoDynBaselineRewardsCfg):
    oscillator_tracking = None
    oscillator_deviation = RewTerm(func=mdp.oscillator_joint_deviation_l2, weight=-0.5)


@configclass
class G1CoDynResidualEnvCfg(G1CoDynBaselineEnvCfg):
    actions: G1CoDynResidualActionsCfg = G1CoDynResidualActionsCfg()
    rewards: G1CoDynResidualRewardsCfg = G1CoDynResidualRewardsCfg()


@configclass
class G1CoDynResidualPlayEnvCfg(G1CoDynBaselinePlayEnvCfg):
    actions: G1CoDynResidualActionsCfg = G1CoDynResidualActionsCfg()
    rewards: G1CoDynResidualRewardsCfg = G1CoDynResidualRewardsCfg()

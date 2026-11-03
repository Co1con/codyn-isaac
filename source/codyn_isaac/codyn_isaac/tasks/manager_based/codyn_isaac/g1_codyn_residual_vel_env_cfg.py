"""Baseline settings with four reference-plus-residual targets and no trajectory tracking reward."""

from isaaclab.utils import configclass

from . import mdp
from .g1_codyn_baseline_vel_env_cfg import G1CoDynEnvCfg, G1CoDynPlayEnvCfg, G1CoDynRewardsCfg


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
class G1CoDynResidualRewardsCfg(G1CoDynRewardsCfg):
    oscillator_tracking = None


@configclass
class G1CoDynResidualEnvCfg(G1CoDynEnvCfg):
    actions: G1CoDynResidualActionsCfg = G1CoDynResidualActionsCfg()
    rewards: G1CoDynResidualRewardsCfg = G1CoDynResidualRewardsCfg()


@configclass
class G1CoDynResidualPlayEnvCfg(G1CoDynPlayEnvCfg):
    actions: G1CoDynResidualActionsCfg = G1CoDynResidualActionsCfg()
    rewards: G1CoDynResidualRewardsCfg = G1CoDynResidualRewardsCfg()

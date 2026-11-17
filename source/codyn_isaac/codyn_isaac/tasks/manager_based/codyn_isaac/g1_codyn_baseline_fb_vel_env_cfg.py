"""Baseline velocity tracking with hip/knee field feedback in the observations."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from .g1_codyn_baseline_vel_env_cfg import (
    G1CoDynBaselineEnvCfg,
    G1CoDynBaselineEventsCfg,
    G1CoDynBaselineObservationsCfg,
    G1CoDynBaselinePlayEnvCfg,
)
from .mdp.feedback_reference import feedback_oscillator_position, reset_feedback_oscillator


@configclass
class G1CoDynBaselineFbObservationsCfg(G1CoDynBaselineObservationsCfg):
    @configclass
    class PolicyCfg(G1CoDynBaselineObservationsCfg.PolicyCfg):
        oscillator_position = ObsTerm(func=feedback_oscillator_position)

    @configclass
    class CriticCfg(G1CoDynBaselineObservationsCfg.CriticCfg):
        oscillator_position = ObsTerm(func=feedback_oscillator_position)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1CoDynBaselineFbEventsCfg(G1CoDynBaselineEventsCfg):
    reset_oscillator = EventTerm(func=reset_feedback_oscillator, mode="reset")


@configclass
class G1CoDynBaselineFbEnvCfg(G1CoDynBaselineEnvCfg):
    oscillator_feedback_half_gain_error_deg: float = 10.0
    observations: G1CoDynBaselineFbObservationsCfg = G1CoDynBaselineFbObservationsCfg()
    events: G1CoDynBaselineFbEventsCfg = G1CoDynBaselineFbEventsCfg()


@configclass
class G1CoDynBaselineFbPlayEnvCfg(G1CoDynBaselinePlayEnvCfg):
    oscillator_feedback_half_gain_error_deg: float = 10.0
    observations: G1CoDynBaselineFbObservationsCfg = G1CoDynBaselineFbObservationsCfg()
    events: G1CoDynBaselineFbEventsCfg = G1CoDynBaselineFbEventsCfg()

"""One-control-step prediction of the frozen hip/knee field from measured angles."""

import math

import torch

from .events import reset_oscillator
from .reference import JOINT_NAMES, GaitReference


class FeedbackGaitReference(GaitReference):
    """Query the coupled field from actual joint angles in Isaac joint order."""

    def __init__(
        self, checkpoint, num_envs, device, substeps=4, coupling_gain=0.25,
        *, gain_limit=2.0, saturation_error_deg=10.0,
    ):
        if not math.isfinite(gain_limit) or gain_limit < 1:
            raise ValueError("gain_limit must be finite and at least one")
        if not math.isfinite(saturation_error_deg) or saturation_error_deg <= 0:
            raise ValueError("saturation_error_deg must be finite and positive")
        super().__init__(checkpoint, num_envs, device, substeps=substeps, coupling_gain=coupling_gain)
        self.gain_limit = gain_limit
        self.saturation_error_squared = math.radians(saturation_error_deg) ** 2

    def _step_gain(self, state: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        """Return separate gains from each leg's squared hip/knee orbit distance."""
        segments = bank.roll(-1, dims=1) - bank
        offsets = state[:, :, None, :] - bank[:, None, :, :]
        segment_length_squared = segments.square().sum(dim=-1).clamp_min(1e-12)
        fractions = (offsets * segments[:, None]).sum(dim=-1) / segment_length_squared[:, None]
        residual = offsets - fractions.clamp(0, 1)[..., None] * segments[:, None]
        leg_distance_squared = residual.square().sum(dim=-1).amin(dim=-1)
        fraction = (leg_distance_squared / self.saturation_error_squared).clamp_max(1)
        return 1 + (self.gain_limit - 1) * fraction

    @torch.no_grad()
    def predict(self, joint_position: torch.Tensor, command_speed: torch.Tensor, dt: float) -> torch.Tensor:
        """Return the next hip/knee angles in robot coordinates, without mutating the input."""
        state = (joint_position * self.signs).reshape(-1, 2, 2)
        bank = self._bank_at(command_speed)
        rates = self._coupling_rates(state, bank)
        gain = self._step_gain(state, bank)
        parameters = self.model.parameters_at(command_speed.repeat_interleave(2))
        velocity = self.model.field(state.reshape(-1, 2), parameters).reshape_as(state)
        velocity = velocity * rates[..., None]
        increment = (dt * gain[..., None] * velocity).flatten(1) * self.signs
        return joint_position + increment


def feedback_oscillator_position(env):
    """Cache one measured-state forecast for the policy and critic."""
    if not hasattr(env, "feedback_gait_reference"):
        env.feedback_gait_reference = FeedbackGaitReference(
            env.cfg.oscillator_checkpoint, env.num_envs, env.device,
            coupling_gain=env.cfg.oscillator_coupling_gain,
            gain_limit=env.cfg.oscillator_feedback_gain_limit,
            saturation_error_deg=env.cfg.oscillator_feedback_saturation_error_deg,
        )
        names = env.scene["robot"].data.joint_names
        env.feedback_joint_ids = [names.index(name) for name in JOINT_NAMES]
        env.feedback_reference_step = -1
    if env.feedback_reference_step != env.common_step_counter:
        actual = env.scene["robot"].data.joint_pos[:, env.feedback_joint_ids]
        speed = env.command_manager.get_command("base_velocity")[:, 0]
        env.feedback_reference_position = env.feedback_gait_reference.predict(actual, speed, env.step_dt)
        env.feedback_reference_step = env.common_step_counter
    return env.feedback_reference_position


def reset_feedback_oscillator(env, env_ids):
    """Reset the nominal reward reference and invalidate pre-reset observations."""
    reset_oscillator(env, env_ids)
    env.feedback_reference_step = -1

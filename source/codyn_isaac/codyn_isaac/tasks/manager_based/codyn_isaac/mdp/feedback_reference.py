"""One-control-step prediction of the frozen hip/knee field from measured angles."""

import math

import torch

from .events import reset_oscillator
from .reference import JOINT_NAMES, GaitReference


class FeedbackGaitReference(GaitReference):
    """Query the coupled field from actual joint angles in Isaac joint order."""

    def __init__(self, checkpoint, num_envs, device, substeps=4, coupling_gain=0.25, *, half_gain_error_deg=10.0):
        if not math.isfinite(half_gain_error_deg) or half_gain_error_deg <= 0:
            raise ValueError("half_gain_error_deg must be finite and positive")
        super().__init__(checkpoint, num_envs, device, substeps=substeps, coupling_gain=coupling_gain)
        self.half_gain_error_squared = math.radians(half_gain_error_deg) ** 2

    def _step_gain(self, state: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        """Scale the horizon using the RMS angular distance to both leg orbits."""
        segments = bank.roll(-1, dims=1) - bank
        offsets = state[:, :, None, :] - bank[:, None, :, :]
        segment_length_squared = segments.square().sum(dim=-1).clamp_min(1e-12)
        fractions = (offsets * segments[:, None]).sum(dim=-1) / segment_length_squared[:, None]
        residual = offsets - fractions.clamp(0, 1)[..., None] * segments[:, None]
        leg_distance_squared = residual.square().sum(dim=-1).amin(dim=-1)
        rms_error_squared = leg_distance_squared.sum(dim=-1) / 4
        return (1 + rms_error_squared / self.half_gain_error_squared).reciprocal()

    @torch.no_grad()
    def predict(self, joint_position: torch.Tensor, command_speed: torch.Tensor, dt: float) -> torch.Tensor:
        """Return the next hip/knee angles in robot coordinates, without mutating the input."""
        state = (joint_position * self.signs).reshape(-1, 2, 2)
        bank = self._bank_at(command_speed)
        rates = self._coupling_rates(state, bank).reshape(-1, 1)
        gain = self._step_gain(state, bank)
        parameters = self.model.parameters_at(command_speed.repeat_interleave(2))
        state = state.reshape(-1, 2)
        h = dt / self.substeps * gain.repeat_interleave(2)[:, None]

        def field(q):
            return self.model.field(q, parameters) * rates

        for _ in range(self.substeps):
            a = field(state)
            b = field(state + h * a / 2)
            c = field(state + h * b / 2)
            d = field(state + h * c)
            state = state + h * (a + 2 * b + 2 * c + d) / 6
        return state.reshape(-1, 4) * self.signs


def feedback_oscillator_position(env):
    """Cache one measured-state forecast for the policy and critic."""
    if not hasattr(env, "feedback_gait_reference"):
        env.feedback_gait_reference = FeedbackGaitReference(
            env.cfg.oscillator_checkpoint, env.num_envs, env.device,
            coupling_gain=env.cfg.oscillator_coupling_gain,
            half_gain_error_deg=env.cfg.oscillator_feedback_half_gain_error_deg,
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

"""One-control-step prediction of the frozen hip/knee field from measured angles."""

import torch

from .events import reset_oscillator
from .reference import JOINT_NAMES, GaitReference


class FeedbackGaitReference(GaitReference):
    """Query the coupled field from actual joint angles in Isaac joint order."""

    @torch.no_grad()
    def predict(self, joint_position: torch.Tensor, command_speed: torch.Tensor, dt: float) -> torch.Tensor:
        """Return the next hip/knee angles in robot coordinates, without mutating the input."""
        state = (joint_position * self.signs).reshape(-1, 2, 2)
        rates = self._coupling_rates(state, self._bank_at(command_speed)).reshape(-1, 1)
        parameters = self.model.parameters_at(command_speed.repeat_interleave(2))
        state = state.reshape(-1, 2)
        h = dt / self.substeps

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

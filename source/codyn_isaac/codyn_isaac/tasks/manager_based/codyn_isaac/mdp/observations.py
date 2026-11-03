"""Reference observation shared by rewards and observations; advance once per environment step."""

import torch
from .reference import GaitReference, JOINT_NAMES


def planar_velocity_commands(env, command_name="base_velocity"):
    """Policy/critic command order: forward speed, yaw rate; lateral speed is fixed at zero."""
    return env.command_manager.get_command(command_name)[:, [0, 2]]


def get_reference(env):
    if not hasattr(env, "gait_reference"):
        env.gait_reference = GaitReference(env.cfg.oscillator_checkpoint, env.num_envs, env.device)
        robot = env.scene["robot"]
        env.gait_reference.joint_ids = [robot.data.joint_names.index(name) for name in JOINT_NAMES]
        env.gait_reference_step = env.common_step_counter
        env.gait_reference_reset = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return env.gait_reference


def step_reference(env):
    reference = get_reference(env)
    command = env.command_manager.get_command("base_velocity")[:, 0]
    if env.gait_reference_step != env.common_step_counter:
        reference.step(command, env.step_dt)
        env.gait_reference_step = env.common_step_counter
    # Reset events precede command-manager reset; consume the marker here after commands refresh.
    ids = env.gait_reference_reset.nonzero(as_tuple=False).flatten()
    if ids.numel():
        reference.reset(ids, command[ids])
        env.gait_reference_reset[ids] = False
    return reference


def oscillator_position(env):
    reference = step_reference(env)
    limits = env.scene["robot"].data.soft_joint_pos_limits[:, reference.joint_ids]
    return reference.position.clamp(limits[..., 0], limits[..., 1])



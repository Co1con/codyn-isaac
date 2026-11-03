"""Read-only reference observations: callbacks never advance oscillator state."""

from .reference import GaitReference, JOINT_NAMES


def planar_velocity_commands(env, command_name="base_velocity"):
    """Policy/critic command order: forward speed, yaw rate; lateral speed is fixed at zero."""
    return env.command_manager.get_command(command_name)[:, [0, 2]]


def get_reference(env):
    if not hasattr(env, "gait_reference"):
        env.gait_reference = GaitReference(env.cfg.oscillator_checkpoint, env.num_envs, env.device)
        robot = env.scene["robot"]
        env.gait_reference.joint_ids = [robot.data.joint_names.index(name) for name in JOINT_NAMES]
    return env.gait_reference


def oscillator_position(env):
    reference = get_reference(env)
    limits = env.scene["robot"].data.soft_joint_pos_limits[:, reference.joint_ids]
    return reference.position.clamp(limits[..., 0], limits[..., 1])



"""Reset markers for the neural oscillator; other G1 events stay in env_cfg."""

from .observations import get_reference


def reset_oscillator(env, env_ids):
    get_reference(env)
    env.gait_reference_reset[env_ids] = True

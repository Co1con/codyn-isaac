"""G1 29-DoF velocity tracking with frozen neural-oscillator references."""

import gymnasium as gym

for name, config in (("Train", "G1CoDynEnvCfg"), ("Play", "G1CoDynPlayEnvCfg")):
    gym.register(
        id=f"CoDyn-G1-29DoF-{name}-v0",
        entry_point=f"{__name__}.env:G1CoDynEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_codyn_vel_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1PPORunnerCfg",
        },
    )

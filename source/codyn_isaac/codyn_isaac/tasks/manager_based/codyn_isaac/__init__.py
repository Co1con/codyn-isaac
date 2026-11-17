"""G1 29-DoF velocity tracking tasks."""

import gymnasium as gym

for mode, config in (("Train", "G1BaselineEnvCfg"), ("Play", "G1BaselinePlayEnvCfg")):
    gym.register(
        id=f"G1-29DoF-Baseline-{mode}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_baseline_vel_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1UnitreeBaselinePPORunnerCfg",
        },
    )

for name, config in (("Train", "G1CoDynBaselineEnvCfg"), ("Play", "G1CoDynBaselinePlayEnvCfg")):
    gym.register(
        id=f"CoDyn-G1-29DoF-{name}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_codyn_baseline_vel_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1BaselinePPORunnerCfg",
        },
    )

# Explicit baseline names alongside the original task IDs for existing commands.
for variant, module, configs, runner in (
    ("Baseline", "g1_codyn_baseline_vel_env_cfg", ("G1CoDynBaselineEnvCfg", "G1CoDynBaselinePlayEnvCfg"), "G1BaselinePPORunnerCfg"),
    (
        "Baseline-FB", "g1_codyn_baseline_fb_vel_env_cfg",
        ("G1CoDynBaselineFbEnvCfg", "G1CoDynBaselineFbPlayEnvCfg"), "G1BaselineFbPPORunnerCfg",
    ),
    ("Residual", "g1_codyn_residual_vel_env_cfg", ("G1CoDynResidualEnvCfg", "G1CoDynResidualPlayEnvCfg"), "G1ResidualPPORunnerCfg"),
):
    for mode, config in zip(("Train", "Play"), configs):
        gym.register(
            id=f"CoDyn-G1-29DoF-{variant}-{mode}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.{module}:{config}",
                "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:{runner}",
            },
        )

# CoDyn G1 baseline and residual tasks

The original configuration is now `g1_codyn_baseline_vel_env_cfg.py`.
The original `CoDyn-G1-29DoF-Train-v0` / `Play-v0` task IDs and old config
import path remain compatible. Explicit baseline aliases are
`CoDyn-G1-29DoF-Baseline-Train-v0` / `Baseline-Play-v0`.

`g1_codyn_residual_vel_env_cfg.py` inherits the baseline settings and changes:

- Removes only `oscillator_tracking`. Velocity tracking and all other rewards remain.
- Left/right hip pitch and knee: `target = oscillator_reference + 0.1 * action`.
- Hip/knee scale is fixed at 0.1 rad, with no extra residual action clipping.
  The runner's existing action limit of ±5 gives a maximum correction of ±0.5 rad (±28.65°).
- Other 25 joints: `target = default_joint_position + 0.25 * action`.
- Reference uses the existing hip signs, soft-limit clipping, per-step cache and resets.
- Keeps one 29-dimensional action term in the original joint order, including symmetry augmentation.
- Separates residual logs under `logs/rsl_rl/g1_codyn_residual_velocity/`; PPO hyperparameters are unchanged.

The oscillator remains frozen and autonomous. This action change does not add body
feedback or joint training. All joints retain the runner's existing 5.0 action
clipping. Reference clipping is unchanged, and no extra clipping is introduced
for the summed target. A small scale reduces the angle change for a given action;
it does not prevent PPO from learning larger actions to compensate.

Run from this directory in the existing Isaac Lab environment:

```bash
# First validate environment setup and one short training run.
python scripts/rsl_rl/train.py --task CoDyn-G1-29DoF-Residual-Train-v0 --num_envs 32 --max_iterations 2 --headless
python scripts/rsl_rl/train.py --task CoDyn-G1-29DoF-Residual-Train-v0 --num_envs 4096 --headless
python scripts/rsl_rl/play.py --task CoDyn-G1-29DoF-Residual-Play-v0 --num_envs 1
```

Start a new residual run after changing the action scale: baseline checkpoints
and earlier residual checkpoints using scale 0.5 have different action semantics.

Simulation-free checks (PyTorch required):

```bash
python scripts/check_reference.py --device cpu --num_envs 16 --steps 10
python scripts/check_residual_action.py --isaaclab-root /path/to/IsaacLab
```

These checks do not validate robot physics or PPO convergence.

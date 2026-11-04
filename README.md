# CoDyn G1 baseline and residual tasks

The original configuration is now `g1_codyn_baseline_vel_env_cfg.py`.
The original `CoDyn-G1-29DoF-Train-v0` / `Play-v0` task IDs remain compatible.
Baseline configuration classes now use the `G1CoDynBaseline...Cfg` names, including
`G1CoDynBaselineEnvCfg` and `G1CoDynBaselinePlayEnvCfg`; the runner is
`G1BaselinePPORunnerCfg`. Explicit baseline aliases are
`CoDyn-G1-29DoF-Baseline-Train-v0` / `Baseline-Play-v0`.

`g1_codyn_residual_vel_env_cfg.py` inherits the baseline settings and changes:

- Replaces the positive `oscillator_tracking` reward with a weak quadratic penalty:
  `oscillator_deviation = -0.2 * sum((actual_hip_knee - oscillator_reference)**2)`.
  Angles use radians and the four joints are left/right hip pitch and knee.
  Velocity tracking and other rewards are inherited from the baseline configuration.
- Left/right hip pitch and knee: `target = oscillator_reference + 0.05 * action`.
- Hip/knee scale is fixed at 0.05 rad, with no extra residual action clipping.
  The runner's existing action limit of ±5 gives a maximum correction of ±0.25 rad (±14.32°).
- Other 25 joints: `target = default_joint_position + 0.25 * action`.
- Reference uses the existing hip signs, per-step cache and resets, without angle clipping.
- Keeps one 29-dimensional action term in the original joint order, including symmetry augmentation.
- Separates residual logs under `logs/rsl_rl/g1_codyn_residual_velocity/`; PPO hyperparameters are unchanged.

The oscillator remains frozen and autonomous. This action change does not add body
feedback or joint training. All joints retain the runner's existing 5.0 action
clipping. Neither the oscillator reference nor the summed target is clipped to soft
joint limits. A small scale reduces the angle change for a given action;
it does not prevent PPO from learning larger actions to compensate.

Both tasks now use bidirectional anti-phase coordination with
`oscillator_coupling_gain=0.25`. Phase is estimated by projection onto the frozen
model's speed-conditioned, uniformly time-sampled cycle bank. With
`e = phase_right - phase_left - pi`, the left field is multiplied by
`1 + gain*sin(e)` and the right field by `1 - gain*sin(e)`. Rates are held for one
control interval and remain in [0.75, 1.25]. At constant commanded speed this changes
progression along the original field without changing its state-space orbits.
Angles are never clipped or forced to be negatives of the other leg.
The sinusoidal coupling has an unstable in-phase equilibrium; it does not break
perfectly identical left/right states by itself. Reset still initializes half-cycle offsets.
Set the gain to 0 for an uncoupled ablation. No robot-state feedback is introduced yet.

Both tasks penalize waist pitch/roll deviation with `joint_deviation_l1` at weight
-2.0, and waist yaw deviation separately at -0.2. Existing exported policies retain
their old learned behavior; retraining or fine-tuning is required to learn these rewards.
Policies trained before reference clipping was removed saw a different reference observation.

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
python scripts/check_coupling.py
```

These checks do not validate robot physics or PPO convergence.
`check_coupling.py` tests initial phase errors and a right-leg phase disturbance at
constant/changing speeds, and saves diagnostics under `log/codyn-isaac/reference_coupling`.

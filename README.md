# CoDyn Isaac Lab — G1 training

Isaac Lab v2.3.2 external template, Manager-Based workflow, RSL-RL/PPO.

## Structure

```text
source/codyn_isaac/codyn_isaac/
  assets/G1/                         # G1 robot assets
  assets/oscillator/M01_left.pt       # frozen model weights and initialization cycle bank
  tasks/manager_based/codyn_isaac/
    g1_codyn_vel_env_cfg.py           # original G1 configuration, HALO terms replaced
    agents/rsl_rl_ppo_cfg.py
    mdp/observations.py              # reference inference, cached once per environment step
    mdp/events.py                    # reference reset event
    mdp/
```

The original G1 actions, rewards unrelated to HALO, events and terminations are retained.
Tasks use Isaac Lab's standard `ManagerBasedRLEnv`. Reference rewards and observations
share one oscillator state; reset events mark the environments to reinitialize after
their velocity commands refresh. The required four G1 29-DoF USD files are included
in Git, overriding the template's default USD ignore rule.
The active velocity commands are forward speed and yaw rate. The actor/critic receive
`[lin_vel_x, ang_vel_z]`; lateral speed is always zero. Isaac Lab retains its standard
three-slot velocity representation internally for rewards and visualization.
Forward speed starts at 0.8–1.0 and expands to 0.8–1.6; yaw rate starts at
-0.3–0.3 and expands to -0.5–0.5. Standing commands are disabled. The learned oscillator replaces the HALO reference
observation and hip/knee tracking reward. Hip polarity follows the original mapping:
`[-left_hip, left_knee, -right_hip, right_knee]`. The tracking target is limited to
G1 soft joint limits. PPO still outputs all 29 joint position actions.

The two legs share the frozen M01-left model and start half a cycle apart. Their
states evolve independently; no additional phase coupling or speed filter is added.
The model receives the original commanded speed directly. The forward command remains within the trained 0.8–1.6 speed-label range.
Turning is learned by PPO; the reference model itself is not yaw-conditioned.

## Linux / remote GPU

Use an existing Isaac Lab v2.3.2 environment with its RSL-RL dependencies installed.
From this repository root:

```bash
python -m pip install -e source/codyn_isaac
python scripts/list_envs.py
python scripts/check_reference.py --device cuda --num_envs 1024
python scripts/rsl_rl/train.py --task CoDyn-G1-29DoF-Train-v0 --num_envs 32 --max_iterations 2 --headless
python scripts/rsl_rl/train.py --task CoDyn-G1-29DoF-Train-v0 --num_envs 4096 --headless
python scripts/rsl_rl/play.py --task CoDyn-G1-29DoF-Play-v0 --num_envs 1
```

`check_reference.py` checks the PyTorch reference core, per-step caching, subset resets and observation
symmetry without starting Isaac Sim. It does not validate robot physics or PPO.
Run the 32-environment smoke test on the remote machine before full training.
Training logs retain the official `logs/rsl_rl/g1_codyn_velocity/` structure.

To refresh the frozen oscillator from an offline training checkpoint:

```bash
python scripts/export_oscillator.py --checkpoint /path/to/model.pt
```

The export script uses NumPy, SciPy and PyTorch and regenerates the phase bank from
the model itself. The deployed training package only needs PyTorch for inference.

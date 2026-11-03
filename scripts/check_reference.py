"""Test frozen reference inference/reset/symmetry without launching Isaac Sim.

python scripts/check_reference.py --device cuda --num_envs 1024 --steps 100
"""

import argparse
import ast
import importlib
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "source/codyn_isaac/codyn_isaac"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.manual_seed(42)
    mdp_path = PACKAGE / "tasks/manager_based/codyn_isaac/mdp"
    # Load only the standalone PyTorch core, avoiding Isaac Lab package initialization.
    package = types.ModuleType("codyn_reference_check"); package.__path__ = [str(mdp_path)]
    sys.modules[package.__name__] = package
    module = importlib.import_module("codyn_reference_check.reference")
    reference = module.GaitReference(PACKAGE / "assets/oscillator/M01_left.pt", args.num_envs, args.device)
    assert not any(p.requires_grad for p in reference.model.parameters())
    original = reference.state.clone()
    reference.reset(torch.tensor([0, 1], device=args.device), torch.tensor([0.8, 1.6], device=args.device))
    torch.testing.assert_close(reference.state[2:], original[2:], rtol=0, atol=0)
    for i in range(args.steps):
        speeds = torch.linspace(0.8, 1.6, args.num_envs, device=args.device)
        if i % 20 >= 10:
            speeds = speeds.flip(0)
        reference.step(speeds, 0.02)
    assert torch.isfinite(reference.state).all() and torch.isfinite(reference.velocity).all()
    torch.testing.assert_close(reference.position, reference.state.flatten(1) * reference.signs)
    # Test the exact migrated symmetry transforms without requiring TensorDict/Isaac.
    tree = ast.parse((mdp_path / "symmetry.py").read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom)) or isinstance(n, ast.ImportFrom) and n.module == '__future__']
    namespace = {"torch": torch}
    exec(compile(tree, str(mdp_path / "symmetry.py"), 'exec'), namespace)
    source = ast.parse((PACKAGE / "assets/G1/unitree.py").read_text(encoding="utf-8"))
    assignment = next(n for n in source.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'UNITREE_G1_29DOF_CFG' for t in n.targets))
    names = tuple(ast.literal_eval(next(k.value for k in assignment.value.keywords if k.arg == 'joint_sdk_names')))
    # Exercise manager callbacks without Isaac: repeated reward/observation reads,
    # then a partial episode reset after command resampling in the same step.
    observations = importlib.import_module("codyn_reference_check.observations")
    events = importlib.import_module("codyn_reference_check.events")
    commands = torch.zeros(args.num_envs, 3, device=args.device); commands[:, 0] = 1.0
    limits = torch.tensor([-10.0, 10.0], device=args.device).expand(args.num_envs, len(names), 2)
    env = types.SimpleNamespace(
        cfg=types.SimpleNamespace(oscillator_checkpoint=PACKAGE / "assets/oscillator/M01_left.pt"),
        num_envs=args.num_envs, device=args.device, common_step_counter=0, step_dt=0.02,
        command_manager=types.SimpleNamespace(get_command=lambda name: commands),
        scene={"robot": types.SimpleNamespace(data=types.SimpleNamespace(joint_names=names, soft_joint_pos_limits=limits))},
    )
    observations.oscillator_position(env)
    for step in range(1, 4):
        env.common_step_counter = step
        observations.oscillator_position(env)
        state = env.gait_reference.state.clone()
        observations.oscillator_position(env)
        torch.testing.assert_close(env.gait_reference.state, state, rtol=0, atol=0)
    ids = torch.tensor([0, 1], device=args.device)
    events.reset_oscillator(env, ids)
    commands[ids, 0] = 1.6
    observations.oscillator_position(env)
    torch.testing.assert_close(env.gait_reference.state[2:], state[2:], rtol=0, atol=0)
    torch.testing.assert_close(env.gait_reference.speed[ids], commands[ids, 0])
    assert not env.gait_reference_reset.any()
    state = env.gait_reference.state.clone()
    observations.oscillator_position(env)
    torch.testing.assert_close(env.gait_reference.state, state, rtol=0, atol=0)
    obs = torch.randn(4, 128, device=args.device)
    transform = namespace['_transform_g1_policy_obs_left_right']
    mirrored = transform(obs, names)
    torch.testing.assert_close(transform(mirrored, names), obs)
    torch.testing.assert_close(mirrored[:, 6], obs[:, 6])
    torch.testing.assert_close(mirrored[:, 7], -obs[:, 7])
    torch.testing.assert_close(mirrored[:, -4:], obs[:, -4:][:, [2, 3, 0, 1]])
    print(f"PASS: {args.num_envs} environments, {args.steps} steps, {args.device}; frozen weights, subset reset, finite dynamics, joint mapping, per-step cache, reset commands, symmetry.")


if __name__ == "__main__":
    main()

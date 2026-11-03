"""Check residual targets against Isaac Lab's actual joint action implementation, without Isaac Sim.

python scripts/check_residual_action.py --isaaclab-root /path/to/IsaacLab
Only the scene/manager shell is replaced; JointAction, JointPositionAction and
the regex resolver are loaded from the supplied Isaac Lab source.
"""

import argparse
import ast
import importlib.util
import types
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaaclab-root", type=Path, required=True)
    args = parser.parse_args()
    lab = args.isaaclab_root / "source/isaaclab/isaaclab"
    mdp = Path(__file__).resolve().parents[1] / "source/codyn_isaac/codyn_isaac/tasks/manager_based/codyn_isaac/mdp"
    spec = importlib.util.spec_from_file_location("lab_strings", lab / "utils/string.py")
    strings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strings)

    class ActionTerm:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env
            self._asset = env.scene[cfg.asset_name]
            self.num_envs, self.device = env.num_envs, env.device

    namespace = {"torch": torch, "ActionTerm": ActionTerm, "string_utils": strings,
                 "omni": types.SimpleNamespace(log=types.SimpleNamespace(info=lambda message: None))}

    def load_classes(file, names):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + [
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names
        ]
        exec(compile(ast.fix_missing_locations(tree), str(file), "exec"), namespace)

    load_classes(lab / "envs/mdp/actions/joint_actions.py", {"JointAction", "JointPositionAction"})
    reference_names = ("left_hip_pitch_joint", "left_knee_joint", "right_hip_pitch_joint", "right_knee_joint")
    namespace.update(JOINT_NAMES=reference_names, oscillator_position=lambda env: env.reference)
    load_classes(mdp / "actions.py", {"OscillatorResidualJointPositionAction"})
    # Use shuffled joint order and nonzero offsets to expose index/offset mistakes.
    names = ["other_0", reference_names[2], "other_1", reference_names[0], reference_names[3], reference_names[1]]
    names += [f"other_{i}" for i in range(2, 25)]
    defaults = torch.linspace(-0.4, 0.7, 29).expand(3, -1).clone()
    written = []
    asset = types.SimpleNamespace(
        num_joints=29,
        data=types.SimpleNamespace(default_joint_pos=defaults),
        find_joints=lambda patterns, preserve_order: (list(range(29)), names),
        set_joint_position_target=lambda targets, joint_ids: written.append(targets.clone()),
    )
    env = types.SimpleNamespace(num_envs=3, device="cpu", scene={"robot": asset},
                                reference=torch.tensor([[-0.2, 0.6, 0.1, 0.8]]).expand(3, -1).clone())
    # Read the production scale dictionary, rather than recreating its regexes.
    tree = ast.parse((mdp.parent / "g1_codyn_residual_vel_env_cfg.py").read_text(encoding="utf-8"))
    action_cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1CoDynResidualActionsCfg")
    call = action_cfg.body[0].value
    scales = ast.literal_eval(next(kw.value for kw in call.keywords if kw.arg == "scale"))
    cfg = types.SimpleNamespace(asset_name="robot", joint_names=[".*"], preserve_order=False,
                                scale=scales, offset=0.0, clip=None, use_default_offset=True)
    baseline = namespace["JointPositionAction"](cfg, env)
    residual = namespace["OscillatorResidualJointPositionAction"](cfg, env)
    ids = [names.index(name) for name in reference_names]
    others = [i for i in range(29) if i not in ids]
    assert residual.action_dim == 29
    torch.testing.assert_close(residual._scale[:, ids], torch.full((3, 4), 0.5))
    torch.testing.assert_close(residual._scale[:, others], torch.full((3, 25), 0.25))
    torch.manual_seed(42)
    for actions in (torch.zeros(3, 29), torch.randn(3, 29), torch.full((3, 29), 5.0)):
        baseline.process_actions(actions)
        reference = env.reference.clone()
        residual.process_actions(actions)
        torch.testing.assert_close(residual.processed_actions[:, ids], reference + 0.5 * actions[:, ids])
        torch.testing.assert_close(residual.processed_actions[:, others], baseline.processed_actions[:, others])
        torch.testing.assert_close(env.reference, reference, rtol=0, atol=0)
        torch.testing.assert_close(residual.raw_actions, actions)
        for _ in range(4):
            residual.apply_actions()
        for target in written[-4:]:
            torch.testing.assert_close(target, residual.processed_actions)
    # Changing reference must replace, rather than accumulate on, the previous target.
    env.reference[0] += 0.3
    residual.process_actions(torch.zeros(3, 29))
    torch.testing.assert_close(residual.processed_actions[:, ids], env.reference)
    residual.reset([0])
    torch.testing.assert_close(residual.raw_actions[0], torch.zeros(29))
    print("PASS: 29-joint order, scale regexes, four residual targets, 25 unchanged joints, no double offset, no accumulation, held physics targets, reset.")


if __name__ == "__main__":
    main()

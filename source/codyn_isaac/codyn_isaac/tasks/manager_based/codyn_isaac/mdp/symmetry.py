"""G1 symmetry augmentation; policy observation dimension is 128."""
from __future__ import annotations
import torch
from tensordict import TensorDict

G1_JOINT_DIM = 29
G1_POLICY_OBS_DIM = 128


@torch.no_grad()
def compute_g1_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Return original and left-right mirrored G1 observations/actions."""
    joint_names = tuple(env.unwrapped.scene["robot"].data.joint_names)
    if len(joint_names) != G1_JOINT_DIM:
        raise ValueError(f"Expected {G1_JOINT_DIM} G1 joints, got {len(joint_names)}.")

    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_g1_policy_obs_left_right(
            obs["policy"][:], joint_names
        )
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device, dtype=actions.dtype)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size : 2 * batch_size] = _transform_g1_actions_left_right(actions, joint_names)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _transform_g1_policy_obs_left_right(obs: torch.Tensor, joint_names: tuple[str, ...]) -> torch.Tensor:
    """Mirror the current 128-D G1 policy observation layout."""
    obs = obs.clone()
    if obs.shape[-1] != G1_POLICY_OBS_DIM:
        raise ValueError(f"Expected G1 policy obs dim {G1_POLICY_OBS_DIM}, got {obs.shape[-1]}.")

    obs[:, 0:3] *= torch.tensor([-1.0, 1.0, -1.0], device=obs.device, dtype=obs.dtype)
    obs[:, 3:6] *= torch.tensor([1.0, -1.0, 1.0], device=obs.device, dtype=obs.dtype)
    obs[:, 6:8] *= torch.tensor([1.0, -1.0], device=obs.device, dtype=obs.dtype)
    end_idx = 8
    for _ in range(4):
        start_idx, end_idx = end_idx, end_idx + G1_JOINT_DIM
        obs[:, start_idx:end_idx] = _switch_g1_29dof_joints_left_right(obs[:, start_idx:end_idx], joint_names)
    obs[:, end_idx:end_idx + 4] = obs[:, end_idx:end_idx + 4][:, [2, 3, 0, 1]]
    return obs


def _transform_g1_actions_left_right(actions: torch.Tensor, joint_names: tuple[str, ...]) -> torch.Tensor:
    return _switch_g1_29dof_joints_left_right(actions, joint_names)


def _switch_g1_29dof_joints_left_right(
    joint_data: torch.Tensor, joint_names: tuple[str, ...]
) -> torch.Tensor:
    if joint_data.shape[-1] != G1_JOINT_DIM:
        raise ValueError(f"Expected G1 joint/action dim {G1_JOINT_DIM}, got {joint_data.shape[-1]}.")
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    mirrored_names = [
        name.replace("left_", "right_", 1)
        if name.startswith("left_")
        else name.replace("right_", "left_", 1)
        if name.startswith("right_")
        else name
        for name in joint_names
    ]
    missing_names = [name for name in mirrored_names if name not in name_to_index]
    if missing_names:
        raise ValueError(f"Missing mirrored G1 joints: {missing_names}.")
    perm = torch.tensor([name_to_index[name] for name in mirrored_names], device=joint_data.device)
    signs = torch.tensor(
        [-1.0 if "_roll_joint" in name or "_yaw_joint" in name else 1.0 for name in joint_names],
        device=joint_data.device,
        dtype=joint_data.dtype,
    )
    return joint_data[..., perm] * signs

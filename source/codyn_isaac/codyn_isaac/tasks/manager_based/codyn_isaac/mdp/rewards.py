"""Migrated G1 rewards and neural-oscillator reference tracking."""
from __future__ import annotations
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from .observations import oscillator_position

def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward xy velocity tracking in the gravity-aligned yaw frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    velocity_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - velocity_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return reward

def energy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)

def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize foot horizontal velocity while the foot is in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(contacts * torch.sum(torch.square(feet_vel_xy), dim=-1), dim=1)

def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground."""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)

def contralateral_shoulder_coordination(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    direction_smoothing: float = 0.02,
    shoulder_velocity_scale: float = 0.3,
) -> torch.Tensor:
    """连续奖励肩部与异侧髋电机反馈速度同向。"""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_names = asset.data.joint_names
    hip_vel = asset.data.joint_vel[:, [joint_names.index("left_hip_pitch_joint"), joint_names.index("right_hip_pitch_joint")]]
    shoulder_vel = asset.data.joint_vel[:, [joint_names.index("right_shoulder_pitch_joint"), joint_names.index("left_shoulder_pitch_joint")]]
    hip_direction = hip_vel / (hip_vel.abs() + direction_smoothing)
    return torch.tanh(hip_direction * shoulder_vel / shoulder_velocity_scale).mean(dim=1)


def oscillator_joint_tracking(env, asset_cfg=SceneEntityCfg("robot"), command_name="base_velocity",
                              joint_names=("left_hip_pitch_joint", "left_knee_joint", "right_hip_pitch_joint", "right_knee_joint")):
    """Keep the original four-joint reward scale; reference uses robot joint polarity."""
    asset = env.scene[asset_cfg.name]
    joint_ids = [asset.data.joint_names.index(name) for name in joint_names]
    actual = asset.data.joint_pos[:, joint_ids]
    return torch.exp(-(actual - oscillator_position(env)).square() / 0.1).sum(dim=1)

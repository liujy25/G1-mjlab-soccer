"""Soccer task MDP terms.

Self-contained MDP functions for observations, rewards, terminations, and
reset events.  Mirrors the mjlab.tasks.velocity.mdp pattern so that the
soccer task does not depend on the velocity task's MDP module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp import (
  sample_uniform,
  quat_from_euler_xyz,
  quat_mul,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def builtin_sensor(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  return sensor.data


def projected_gravity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b


def joint_pos_rel(
  env: ManagerBasedRlEnv,
  biased: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  jnt_ids = asset_cfg.joint_ids
  joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos
  return joint_pos[:, jnt_ids] - default_joint_pos[:, jnt_ids]


def joint_vel_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  jnt_ids = asset_cfg.joint_ids
  return asset.data.joint_vel[:, jnt_ids] - default_joint_vel[:, jnt_ids]


def last_action(
  env: ManagerBasedRlEnv, action_name: str | None = None
) -> torch.Tensor:
  if action_name is None:
    return env.action_manager.action
  return env.action_manager.get_term(action_name).raw_action


def ball_pos_in_robot_frame(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Ball position in robot pelvis frame.

  Returns:
    Tensor of shape (num_envs, 3) — ball (x, y, z) relative to robot pelvis,
    expressed in the pelvis coordinate frame.
  """
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]
  ball_pos_w = ball.data.root_link_pos_w
  robot_pos_w = robot.data.root_link_pos_w
  robot_quat_w = robot.data.root_link_quat_w
  delta_w = ball_pos_w - robot_pos_w
  return quat_apply_inverse(robot_quat_w, delta_w)


def ball_vel_in_robot_frame(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Ball linear velocity in robot pelvis frame.

  Returns:
    Tensor of shape (num_envs, 3).
  """
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]
  ball_vel_w = ball.data.root_link_lin_vel_w
  robot_quat_w = robot.data.root_link_quat_w
  return quat_apply_inverse(robot_quat_w, ball_vel_w)


def motion_ref_joint_pos(
  env: ManagerBasedRlEnv,
  command_name: str = "motion_command",
) -> torch.Tensor:
  """Reference joint positions from motion dataset (O^ref_t part).

  Reads the current frame's reference joint positions from the
  MotionCommand attached to the environment.

  Returns:
    Tensor of shape (num_envs, 29) — reference joint positions (rad).
    Returns zeros if no motion command is attached.
  """
  cmd = getattr(env, command_name, None)
  if cmd is None:
    return torch.zeros(env.num_envs, 29, device=env.device)
  return cmd.joint_pos_ref


def motion_ref_joint_vel(
  env: ManagerBasedRlEnv,
  command_name: str = "motion_command",
) -> torch.Tensor:
  """Reference joint velocities from motion dataset (O^ref_t part).

  Reads the current frame's reference joint velocities from the
  MotionCommand attached to the environment.

  Returns:
    Tensor of shape (num_envs, 29) — reference joint velocities (rad/s).
    Returns zeros if no motion command is attached.
  """
  cmd = getattr(env, command_name, None)
  if cmd is None:
    return torch.zeros(env.num_envs, 29, device=env.device)
  return cmd.joint_vel_ref


def motion_ref_anchor_ang_vel(
  env: ManagerBasedRlEnv,
  command_name: str = "motion_command",
) -> torch.Tensor:
  """Reference anchor angular velocity from motion dataset (O^ref_t part).

  Corresponds to the motion_anchor_ang_vel term in the HumanoidSoccer paper's
  observation space (Section III-A, o^ref_t includes root angular velocity).

  Returns:
    Tensor of shape (num_envs, 3) — reference anchor angular velocity (rad/s).
    Returns zeros if no motion command is attached.
  """
  cmd = getattr(env, command_name, None)
  if cmd is None:
    return torch.zeros(env.num_envs, 3, device=env.device)
  return cmd.anchor_ang_vel_ref


def world_point_in_robot_frame(
  env: ManagerBasedRlEnv,
  point: tuple[float, float, float],
  robot_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Convert a fixed world-frame point to robot pelvis frame.

  Args:
    env: The environment.
    point: (x, y, z) in world frame.
    robot_cfg: Robot entity configuration.

  Returns:
    Tensor of shape (num_envs, 3).
  """
  robot: Entity = env.scene[robot_cfg.name]
  robot_pos_w = robot.data.root_link_pos_w
  robot_quat_w = robot.data.root_link_quat_w
  point_t = torch.tensor(point, device=env.device, dtype=torch.float32)
  delta_w = point_t.unsqueeze(0) - robot_pos_w
  return quat_apply_inverse(robot_quat_w, delta_w)


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def is_terminated(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.termination_manager.terminated.float()


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.episode_length_buf >= env.max_episode_length


def bad_orientation(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity_ = asset.data.projected_gravity_b
  return torch.acos(-projected_gravity_[:, 2]).abs() > limit_angle


def bad_anchor_pos_z(
  env: ManagerBasedRlEnv,
  threshold: float = 0.25,
  command_name: str = "motion_command",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when anchor (torso) height deviates too far from motion reference.

  Paper: |z_robot - z_ref| > 0.25m.  Returns False if no motion command.
  """
  cmd = getattr(env, command_name, None)
  if cmd is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  asset: Entity = env.scene[asset_cfg.name]
  torso_idx = asset.body_names.index("torso_link")
  robot_z = asset.data.body_link_pos_w[:, torso_idx, 2]
  ref_z = cmd.anchor_pos_w_ref[:, 2]
  return torch.abs(robot_z - ref_z) > threshold


def bad_anchor_ori(
  env: ManagerBasedRlEnv,
  threshold: float = 0.8,
  command_name: str = "motion_command",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when anchor (torso) orientation deviates too far from motion reference.

  Paper: quat_error > 0.8.  Uses 2*asin(norm(q_robot - q_ref)).
  Returns False if no motion command.
  """
  cmd = getattr(env, command_name, None)
  if cmd is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  asset: Entity = env.scene[asset_cfg.name]
  torso_idx = asset.body_names.index("torso_link")
  robot_q = asset.data.body_link_quat_w[:, torso_idx, :]
  ref_q = cmd.anchor_quat_w_ref
  delta = robot_q - ref_q
  delta_norm = torch.norm(delta, dim=-1).clamp(max=1.0)
  error = 2.0 * torch.asin(delta_norm)
  return error > threshold


def bad_ee_body_pos_z(
  env: ManagerBasedRlEnv,
  threshold: float = 0.25,
  command_name: str = "motion_command",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when any end-effector z deviates too far from motion reference.

  Paper: |z_ee - z_ref| > 0.25m for ankles and wrists.
  Returns False if no motion command.
  """
  cmd = getattr(env, command_name, None)
  if cmd is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  asset: Entity = env.scene[asset_cfg.name]
  ee_names = (
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
  )
  ee_robot_indices = [asset.body_names.index(n) for n in ee_names]

  # Full-body indices for the same end-effectors in the motion data (30 bodies).
  ee_motion_indices = [6, 12, 22, 29]

  robot_z = asset.data.body_link_pos_w[:, ee_robot_indices, 2]  # (E, 4)
  ref_z = torch.cat(
    [cmd.get_ee_pos_w_ref(bi)[:, 2:3] for bi in ee_motion_indices], dim=-1
  )  # (E, 4)
  deviations = torch.abs(robot_z - ref_z)
  return deviations.max(dim=-1).values > threshold


# ---------------------------------------------------------------------------
# Reset events
# ---------------------------------------------------------------------------


def reset_root_state_uniform(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[asset_cfg.name]

  # Pose.
  range_list = [
    pose_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  pose_samples = sample_uniform(
    ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
  )

  # Fixed-base mocap entity path.
  if asset.is_fixed_base:
    if not asset.is_mocap:
      raise ValueError(
        f"Cannot reset root state for fixed-base non-mocap entity "
        f"'{asset_cfg.name}'."
      )
    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    root_states = default_root_state[env_ids].clone()
    positions = (
      root_states[:, 0:3]
      + pose_samples[:, 0:3]
      + env.scene.env_origins[env_ids]
    )
    orientations_delta = quat_from_euler_xyz(
      pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = quat_mul(root_states[:, 3:7], orientations_delta)
    asset.write_mocap_pose_to_sim(
      torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
    return

  # Floating-base entity path.
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_states = default_root_state[env_ids].clone()

  positions = (
    root_states[:, 0:3]
    + pose_samples[:, 0:3]
    + env.scene.env_origins[env_ids]
  )
  orientations_delta = quat_from_euler_xyz(
    pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
  )
  orientations = quat_mul(root_states[:, 3:7], orientations_delta)

  if velocity_range is None:
    velocity_range = {}
  range_list = [
    velocity_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  vel_samples = sample_uniform(
    ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
  )
  velocities = root_states[:, 7:13] + vel_samples

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def reset_joints_by_offset(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  position_range: tuple[float, float],
  velocity_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None

  joint_pos = default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
  joint_pos += sample_uniform(*position_range, joint_pos.shape, env.device)
  joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
  joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

  joint_vel = default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()
  joint_vel += sample_uniform(*velocity_range, joint_vel.shape, env.device)

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)

  asset.write_joint_state_to_sim(
    joint_pos.view(len(env_ids), -1),
    joint_vel.view(len(env_ids), -1),
    env_ids=env_ids,
    joint_ids=joint_ids,
  )


# ---------------------------------------------------------------------------
# Domain randomization events
# ---------------------------------------------------------------------------


def push_robot_base(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  vel_xy_range: tuple[float, float],
  vel_z_range: tuple[float, float],
  ang_vel_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Apply random velocity push to the robot base.

  Args:
    env: The environment.
    env_ids: Environment IDs to push. If None, pushes all.
    vel_xy_range: (min, max) linear velocity range for x and y axes (m/s).
    vel_z_range: (min, max) linear velocity range for z axis (m/s).
    ang_vel_range: (min, max) angular velocity range for roll/pitch/yaw (rad/s).
    asset_cfg: Robot entity configuration.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  n = len(env_ids)

  asset: Entity = env.scene[asset_cfg.name]

  # Random linear velocity (xy same range, z separate).
  lin_vel_xy = sample_uniform(
    vel_xy_range[0], vel_xy_range[1], (n, 2), env.device
  )
  lin_vel_z = sample_uniform(
    vel_z_range[0], vel_z_range[1], (n, 1), env.device
  )
  lin_vel = torch.cat([lin_vel_xy, lin_vel_z], dim=-1)

  # Random angular velocity.
  ang_vel = sample_uniform(
    ang_vel_range[0], ang_vel_range[1], (n, 3), env.device
  )

  velocities = torch.cat([lin_vel, ang_vel], dim=-1)
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def perturb_ball_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  vel_range: tuple[float, float],
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
  """Add random perturbation to ball linear velocity.

  Args:
    env: The environment.
    env_ids: Environment IDs to perturb. If None, perturbs all.
    vel_range: (min, max) per-axis additive velocity perturbation (m/s).
    ball_cfg: Ball entity configuration.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  n = len(env_ids)

  asset: Entity = env.scene[ball_cfg.name]
  current_vel = asset.data.root_link_vel_w[env_ids].clone()
  noise = sample_uniform(vel_range[0], vel_range[1], (n, 3), env.device)
  current_vel[:, :3] += noise
  asset.write_root_link_velocity_to_sim(current_vel, env_ids=env_ids)


# ---------------------------------------------------------------------------
# Soccer-specific reset: random ball velocity toward goal
# ---------------------------------------------------------------------------


@dataclass
class GoalkeeperBallVelCfg:
  """Configuration for randomizing goalkeeper ball velocity.

  The ball is launched from the penalty spot toward the goal area.
  Velocity is randomized in speed, yaw direction, and pitch angle.

  Attributes:
    speed_min: Minimum ball speed (m/s).
    speed_max: Maximum ball speed (m/s).
    yaw_spread_deg: Maximum yaw spread angle (degrees) from straight-ahead.
      The ball direction is uniformly sampled within ±yaw_spread_deg
      relative to the +x (toward goal) direction.
    pitch_min_deg: Minimum launch elevation angle (degrees).
    pitch_max_deg: Maximum launch elevation angle (degrees).
  """

  speed_min: float = 2.0
  speed_max: float = 4.5
  yaw_spread_deg: float = 18.0
  pitch_min_deg: float = 3.0
  pitch_max_deg: float = 8.0


def reset_ball_with_goal_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  vel_cfg: GoalkeeperBallVelCfg,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
  """Reset ball position and apply a randomized velocity toward goal.

  Ball position is reset to its default (from init_state).  Velocity is
  sampled with random direction and speed within the configured ranges so
  every episode presents a different shot.

  Args:
      env: The environment.
      env_ids: Environment IDs to reset. If None, resets all.
      vel_cfg: Velocity randomization configuration.
      ball_cfg: Ball entity configuration.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[ball_cfg.name]
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_states = default_root_state[env_ids].clone()

  # Position: reset to default (at penalty spot) + env origins.
  positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]

  # Speed: uniform in [speed_min, speed_max].
  speeds = (
    vel_cfg.speed_min
    + torch.rand(len(env_ids), device=env.device)
    * (vel_cfg.speed_max - vel_cfg.speed_min)
  )

  # Yaw angle: uniform within ±yaw_spread_deg relative to +x (toward goal).
  spread_rad = math.radians(vel_cfg.yaw_spread_deg)
  yaws = (torch.rand(len(env_ids), device=env.device) * 2 - 1) * spread_rad

  # Pitch angle: uniform in [pitch_min_deg, pitch_max_deg].
  pitch_min_rad = math.radians(vel_cfg.pitch_min_deg)
  pitch_max_rad = math.radians(vel_cfg.pitch_max_deg)
  pitches = (
    pitch_min_rad
    + torch.rand(len(env_ids), device=env.device)
    * (pitch_max_rad - pitch_min_rad)
  )

  # Convert spherical to Cartesian: +x toward goal.
  vx = speeds * torch.cos(pitches) * torch.cos(yaws)
  vy = speeds * torch.cos(pitches) * torch.sin(yaws)
  vz = speeds * torch.sin(pitches)

  # Full root velocity: 6-DOF (lin_vel, ang_vel).  Ball has no initial spin.
  ang_vel = torch.zeros(len(env_ids), 3, device=env.device)
  velocities = torch.cat(
    [torch.stack([vx, vy, vz], dim=-1), ang_vel], dim=-1
  )

  # Quaternion unchanged from default.
  orientations = root_states[:, 3:7]

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)

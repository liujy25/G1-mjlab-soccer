"""Soccer-specific reward functions for shooter training.

Port of HumanoidSoccer's soccer/kick rewards to mjlab.
These complement mjlab's built-in motion tracking rewards
(mjlab.tasks.tracking.mdp.rewards) which handle anchor/body tracking.

NOTE: Reward functions accept raw entity/body/joint names rather than
SceneEntityCfg objects to avoid mjlab's config-resolution re-evaluation bug.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_inv

from .shooter_commands import MultiMotionSoccerCommand
from .shooter_kick_detection import KickContactTracker

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.entity import Entity


def _get_kick_tracker(command: MultiMotionSoccerCommand) -> KickContactTracker:
  tracker = getattr(command, "kick_contact_tracker", None)
  if tracker is None:
    raise RuntimeError("MultiMotionSoccerCommand missing kick_contact_tracker")
  return tracker


def _make_foot_cfg(env: ManagerBasedRlEnv, foot_body_names: tuple[str, ...]) -> SceneEntityCfg:
  """Build a SceneEntityCfg for foot bodies (used at runtime, not config-time)."""
  return SceneEntityCfg("robot", body_names=foot_body_names, body_ids=None)


# -- Action regularization ------------------------------------------------------

def action_rate_l2_clip(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize action rate of change, clamped at 100."""
  delta = env.action_manager.action - env.action_manager.prev_action
  return torch.sum(torch.square(delta), dim=1).clamp(max=100.0)


def waist_action_rate_l2_clip(
  env: ManagerBasedRlEnv,
  entity_name: str = "robot",
  joint_names: tuple[str, ...] = (),
) -> torch.Tensor:
  """Penalize waist joint action rate of change."""
  robot: Entity = env.scene[entity_name]
  idx = torch.as_tensor(
    robot.find_joints(joint_names, preserve_order=True)[0],
    device=env.device,
  )
  delta = env.action_manager.action[:, idx] - env.action_manager.prev_action[:, idx]
  return torch.sum(torch.square(delta), dim=1).clamp(max=100.0)


# -- Stabilization rewards ------------------------------------------------------

def foot_distance(
  env: ManagerBasedRlEnv,
  threshold: float,
  std: float,
  entity_name: str = "robot",
  body_names: tuple[str, ...] = (),
) -> torch.Tensor:
  """Reward minimum separation between feet to avoid crossing."""
  robot: Entity = env.scene[entity_name]
  idx = torch.as_tensor(
    robot.find_bodies(body_names, preserve_order=True)[0],
    device=env.device,
  )
  left_pos = robot.data.body_link_pos_w[:, idx[0]]
  right_pos = robot.data.body_link_pos_w[:, idx[1]]
  dist = torch.norm(left_pos - right_pos, dim=1)
  return torch.where(
    dist >= threshold,
    torch.ones_like(dist),
    torch.exp(-((dist / threshold - 1) ** 2) / (std**2)),
  )


def pelvis_orientation(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
  """Penalize pelvis pitch/roll tilt to keep the robot upright."""
  command = env.command_manager.get_term(command_name)
  gravity_vec_w = torch.tensor(
    [[0.0, 0.0, -1.0]], device=env.device
  ).expand(env.num_envs, -1)
  pelvis_proj = quat_apply_inverse(command.robot_pelvis_quat_w, gravity_vec_w)
  return torch.sum(torch.square(pelvis_proj[:, :2]), dim=1)


def undesired_contacts(
  env: ManagerBasedRlEnv,
  sensor_name: str = "contact_forces",
  threshold: float = 1.0,
  excluded_body_names: tuple[str, ...] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  ),
) -> torch.Tensor:
  """Penalize non-foot/non-wrist ground contacts."""
  sensors = env.scene.sensors
  if sensors is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  try:
    sensor = sensors[sensor_name]
  except (KeyError, TypeError):
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  force = sensor.data.force
  if force is None or force.numel() == 0:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  slots = getattr(sensor, "_slots", ())
  field_name = "force" if any(getattr(s, "field_name", None) == "force" for s in slots) else "found"
  primary_names = [s.primary_name for s in slots if getattr(s, "field_name", None) == field_name]
  if not primary_names:
    force_norm = torch.linalg.vector_norm(force.to(env.device), dim=-1)
    return torch.sum((force_norm > threshold).to(torch.float32), dim=-1)

  num_slots = int(getattr(sensor.cfg, "num_slots", 1))
  num_bodies = len(primary_names)
  force_norm = torch.linalg.vector_norm(force.to(env.device), dim=-1)
  force_norm = force_norm.view(env.num_envs, num_bodies, num_slots)

  excluded = set(excluded_body_names)
  penalized_mask = torch.tensor(
    [name not in excluded for name in primary_names],
    device=env.device,
    dtype=torch.bool,
  )
  contact = force_norm.amax(dim=-1) > threshold
  return torch.sum((contact & penalized_mask.unsqueeze(0)).to(torch.float32), dim=-1)


# -- Soccer / kick rewards ------------------------------------------------------

def _get_target_point_world(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  return command.target_point_pos + env.scene.env_origins


def target_point_proximity(
  env: ManagerBasedRlEnv, std: float, command_name: str = "motion",
) -> torch.Tensor:
  """Reward proximity to the ball; freezes value at first kick contact."""
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  tracker = _get_kick_tracker(command)

  base_xy = command.robot_anchor_pos_w[..., :2]
  target_w = _get_target_point_world(env, command_name)
  diff_xy = base_xy - target_w[..., :2]
  error = torch.sum(diff_xy * diff_xy, dim=-1)
  proximity = torch.exp(-error / std**2)

  contact_awarded = tracker.get_contact_awarded()
  frozen = tracker.get_frozen_proximity_reward()

  new_kick = contact_awarded & (frozen == 0.0)
  if torch.any(new_kick):
    ids = torch.nonzero(new_kick, as_tuple=False).squeeze(-1)
    tracker.freeze_proximity_reward(ids, proximity[ids])
    frozen = tracker.get_frozen_proximity_reward()

  return torch.where(contact_awarded, frozen, proximity)


def target_point_contact(
  env: ManagerBasedRlEnv,
  horizontal_force_threshold: float = 10.0,
  command_name: str = "motion",
  ball_sensor_name: str = "ball_robot_contact",
  foot_body_names: tuple[str, ...] = (),
) -> torch.Tensor:
  """One-shot reward for first valid ball contact with correct foot."""
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  tracker = _get_kick_tracker(command)
  event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  if not torch.any(event.new_contact):
    return reward

  reward_scale = torch.zeros_like(reward)
  correct_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  if foot_body_names:
    foot_cfg = _make_foot_cfg(env, foot_body_names)
    foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
    if foot_info.env_ids.numel() > 0:
      valid = foot_info.expected >= 0
      correct = (foot_info.sides == foot_info.expected) & valid
      reward_scale[foot_info.env_ids] = correct.to(reward_scale.dtype)
      correct_mask[foot_info.env_ids] = correct

  tracker.record_expected_success(event.new_contact, correct_mask)
  return event.new_contact.to(reward.dtype) * reward_scale


def sideways_kick(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  ball_sensor_name: str = "ball_robot_contact",
  horizontal_force_threshold: float = 10.0,
  foot_body_names: tuple[str, ...] = (),
) -> torch.Tensor:
  """Reward foot swing along the expected lateral direction at contact."""
  if not foot_body_names:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  tracker = _get_kick_tracker(command)
  event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  if not torch.any(event.new_contact):
    return reward

  foot_cfg = _make_foot_cfg(env, foot_body_names)
  foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
  if foot_info.env_ids.numel() == 0:
    return reward

  robot = command.robot
  arange = torch.arange(len(foot_info.env_ids), device=env.device)
  foot_vel_w = robot.data.body_link_lin_vel_w[foot_info.env_ids][arange, foot_info.body_indices]
  foot_quat_w = robot.data.body_link_quat_w[foot_info.env_ids][arange, foot_info.body_indices]
  vel_local = quat_apply(quat_inv(foot_quat_w), foot_vel_w)
  vel_norm = torch.norm(vel_local, dim=-1)

  expected_leg = foot_info.expected.to(torch.int8)
  desired_sign = torch.where(expected_leg == 0, -1.0, 1.0)
  directional = vel_local[:, 1] * desired_sign
  axis = torch.clamp(directional, min=0.0)
  alignment = torch.where(vel_norm > 1e-6, axis / vel_norm, torch.zeros_like(vel_norm))
  reward[foot_info.env_ids] = alignment.to(reward.dtype)

  valid = expected_leg >= 0
  correct = (foot_info.sides == foot_info.expected) & valid
  if torch.any(~correct):
    reward[foot_info.env_ids[~correct]] = 0.0
  return reward


def _get_or_init_timer(env: ManagerBasedRlEnv, name: str, length: int) -> torch.Tensor:
  timer = getattr(env, name, None)
  if timer is None or timer.shape[0] != env.num_envs:
    timer = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    setattr(env, name, timer)
  return timer.to(device=env.device, dtype=torch.int32)


def _open_timer_window(
  env: ManagerBasedRlEnv,
  timer_name: str,
  command: MultiMotionSoccerCommand,
  ball_sensor_name: str,
  horizontal_force_threshold: float,
  foot_body_names: tuple[str, ...],
  window_size: int,
) -> torch.Tensor:
  """Open a reward window of `window_size` steps on correct-foot contact."""
  timer = _get_or_init_timer(env, timer_name, window_size)
  tracker = _get_kick_tracker(command)
  event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

  if torch.any(event.new_contact) and foot_body_names:
    foot_cfg = _make_foot_cfg(env, foot_body_names)
    foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
    if foot_info.env_ids.numel() > 0:
      valid = foot_info.expected >= 0
      correct = (foot_info.sides == foot_info.expected) & valid
      correct_ids = foot_info.env_ids[correct]
      if correct_ids.numel() > 0:
        timer[correct_ids] = window_size
  return timer


def ball_velocity_direction_alignment(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  std: float = 0.8,
  velocity_threshold: float = 0.5,
  horizontal_force_threshold: float = 10.0,
  ball_sensor_name: str = "ball_robot_contact",
  foot_body_names: tuple[str, ...] = (),
) -> torch.Tensor:
  """Reward ball velocity aligning with target destination after kick."""
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  ball = env.scene["ball"]
  vel = ball.data.root_link_lin_vel_w
  vel_xy = vel[:, :2]
  vel_xy_norm = torch.norm(vel_xy, dim=-1, keepdim=True)
  vel_norm = torch.norm(vel, dim=-1, keepdim=True)

  direction = command.target_destination_pos - command.initial_target_point_pos
  dir_xy = direction[:, :2]
  dir_norm = torch.norm(dir_xy, dim=-1, keepdim=True)

  valid_mask = (
    (vel_norm.squeeze(-1) > velocity_threshold)
    & (vel_xy_norm.squeeze(-1) > 1e-6)
    & (dir_norm.squeeze(-1) > 1e-6)
  )
  avg_angle = torch.tensor(0.0, device=env.device, dtype=torch.float32)
  if torch.any(valid_mask):
    dir_unit_valid = dir_xy[valid_mask] / dir_norm[valid_mask]
    vel_unit_valid = vel_xy[valid_mask] / vel_xy_norm[valid_mask]
    cos_theta_valid = torch.sum(vel_unit_valid * dir_unit_valid, dim=-1).clamp(-1.0, 1.0)
    avg_angle = torch.acos(cos_theta_valid).mean()
  if hasattr(command, "metrics"):
    command.metrics["ball_velocity_dir_alignment_angle"] = torch.full(
      (env.num_envs,), avg_angle.item(), device=env.device, dtype=torch.float32
    )

  timer_name = f"_{command_name}_dir_align_timer"
  timer = _open_timer_window(env, timer_name, command, ball_sensor_name,
                              horizontal_force_threshold, foot_body_names, 5)

  speed_valid = (
    (vel_xy_norm.squeeze(-1) > 1e-6)
    & (dir_norm.squeeze(-1) > 1e-6)
  )
  active = (timer > 0) & speed_valid

  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  if torch.any(active):
    dir_unit = dir_xy[active] / dir_norm[active]
    vel_unit = vel_xy[active] / vel_xy_norm[active]
    cos_theta = torch.sum(vel_unit * dir_unit, dim=-1).clamp(-1.0, 1.0)
    error = torch.acos(cos_theta) ** 2
    reward[active] = torch.exp(-error / (std**2))

  timer = torch.where(timer > 0, timer - 1, timer)
  setattr(env, timer_name, timer)
  return reward


def ball_speed_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  std: float = 1.2,
  velocity_threshold: float = 0.5,
  horizontal_force_threshold: float = 10.0,
  ball_sensor_name: str = "ball_robot_contact",
  foot_body_names: tuple[str, ...] = (),
) -> torch.Tensor:
  """Reward ball speed magnitude after kick."""
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  ball = env.scene["ball"]
  speed_xy = torch.norm(ball.data.root_link_lin_vel_w[:, :2], dim=-1)

  timer_name = f"_{command_name}_speed_timer"
  timer = _open_timer_window(env, timer_name, command, ball_sensor_name,
                              horizontal_force_threshold, foot_body_names, 5)

  speed_valid = speed_xy > 1e-6
  active = (timer > 0) & speed_valid

  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  if torch.any(active):
    reward[active] = 1.0 - torch.exp(-(speed_xy[active] ** 2) / (std**2))

  timer = torch.where(timer > 0, timer - 1, timer)
  setattr(env, timer_name, timer)
  return reward


def ball_z_speed_penalty_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  std: float = 3.0,
  velocity_threshold: float = 0.5,
  horizontal_force_threshold: float = 10.0,
  ball_sensor_name: str = "ball_robot_contact",
  foot_body_names: tuple[str, ...] = (),
) -> torch.Tensor:
  """Penalty magnitude for excessive vertical ball speed after ball motion starts."""
  del horizontal_force_threshold, ball_sensor_name, foot_body_names
  ball = env.scene["ball"]
  vel = ball.data.root_link_lin_vel_w
  speed = torch.norm(vel, dim=-1)
  z_speed = vel[:, 2]
  valid_mask = speed > velocity_threshold

  timer_name = f"_{command_name}_z_speed_timer"
  prev_name = f"_{command_name}_z_speed_prev"

  timer = getattr(env, timer_name, None)
  if timer is None or timer.shape[0] != env.num_envs:
    timer = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
  else:
    timer = timer.to(device=env.device, dtype=torch.int32)

  prev_valid = getattr(env, prev_name, None)
  if prev_valid is None or prev_valid.shape[0] != env.num_envs:
    prev_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  else:
    prev_valid = prev_valid.to(device=env.device, dtype=torch.bool)

  rising_mask = valid_mask & (~prev_valid)
  timer[rising_mask] = 5
  active = timer > 0
  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  if torch.any(active):
    scale = std if std > 0 else 1.0
    reward[active] = torch.tanh(torch.abs(z_speed[active]) / (scale + 1e-8))

  timer = torch.where(timer > 0, timer - 1, timer)
  setattr(env, timer_name, timer)
  setattr(env, prev_name, valid_mask.to(dtype=torch.bool))
  return reward

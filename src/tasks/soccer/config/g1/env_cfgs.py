"""Unitree G1 soccer environment configurations."""

from dataclasses import replace

from src.assets.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from src.assets.robots.unitree_g1.g1_constants import (
  HOME_KEYFRAME,
  FULL_COLLISION,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from src.tasks.soccer.soccer_env_cfg import make_soccer_env_cfg

import math


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
  """Convert yaw angle (rotation about z-axis) to quaternion (w, x, y, z)."""
  half = yaw / 2.0
  return (math.cos(half), 0.0, 0.0, math.sin(half))


def _g1_robot_at(
  pos: tuple[float, float, float],
  yaw: float = 0.0,
) -> EntityCfg:
  """Create G1 robot entity config at a specific position and yaw."""
  cfg = get_g1_robot_cfg()
  cfg.init_state = replace(HOME_KEYFRAME, pos=pos, rot=_yaw_to_quat(yaw))
  cfg.collisions = (FULL_COLLISION,)
  return cfg


def _setup_robot_env(cfg: ManagerBasedRlEnvCfg) -> None:
  """Apply common G1-specific overrides to a soccer env config."""
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )


def unitree_g1_shooter_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Unitree G1 naive shooter: robot at penalty spot facing goal, ball in front."""
  cfg = make_soccer_env_cfg()

  robot_pos = (-6.2, 0.0, 0.8)
  robot_yaw = 0.0  # Facing +x toward goal
  cfg.scene.entities["robot"] = _g1_robot_at(robot_pos, robot_yaw)
  cfg.scene.entities["ball"].init_state.pos = (-6.0, 0.0, 0.11)

  _setup_robot_env(cfg)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg


def unitree_g1_goalkeeper_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Unitree G1 naive goalkeeper: robot at goal line facing incoming ball."""
  cfg = make_soccer_env_cfg()

  robot_pos = (0.0, 0.0, 0.8)
  robot_yaw = math.pi  # Facing -x toward ball/penalty spot
  cfg.scene.entities["robot"] = _g1_robot_at(robot_pos, robot_yaw)
  cfg.scene.entities["ball"].init_state.pos = (-6.0, 0.0, 0.11)
  # Initial velocity 1 m/s toward goal center (ball travels ~6s to reach goal).
  cfg.scene.entities["ball"].init_state.lin_vel = (0.99, 0.0, 0.13)

  _setup_robot_env(cfg)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg

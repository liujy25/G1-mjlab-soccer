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
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from src.tasks.soccer.soccer_env_cfg import make_soccer_env_cfg
from src.tasks.soccer.config.soccer_settings import SETTINGS
from src.tasks.soccer.mdp import (
  GoalkeeperBallVelCfg,
  reset_ball_with_goal_velocity,
)

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
    secondary=ContactMatch(mode="subtree", pattern="ground", entity="ground"),
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

  s = SETTINGS.scene
  cfg.scene.entities["robot"] = _g1_robot_at(tuple(s.shooter_pos), 0.0)
  cfg.scene.entities["ball"].init_state.pos = tuple(s.ball_pos)

  _setup_robot_env(cfg)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg


def unitree_g1_goalkeeper_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Unitree G1 naive goalkeeper: robot at goal line facing incoming ball.

  Ball velocity is randomized each reset.  Yaw spread is computed from
  goal dimensions and ball distance so shots stay within the goal frame
  (minus goal_margin on each side).
  """
  cfg = make_soccer_env_cfg()
  cfg.episode_length_s = SETTINGS.goalkeeper_episode_length_s

  s = SETTINGS.scene
  cfg.scene.entities["robot"] = _g1_robot_at(tuple(s.goalkeeper_pos), math.pi)
  cfg.scene.entities["ball"].init_state.pos = tuple(s.ball_pos)

  gv = SETTINGS.goalkeeper_ball_vel
  penalty_dist = abs(SETTINGS.scene.ball_pos[0])  # distance to goal line

  # Yaw spread: the ball direction should stay within the goal frame.
  effective_half_width = SETTINGS.goal.width / 2 - gv.goal_margin
  yaw_spread_deg = math.degrees(math.atan(effective_half_width / penalty_dist))

  cfg.events["reset_ball"] = EventTermCfg(
    func=reset_ball_with_goal_velocity,
    mode="reset",
    params={
      "vel_cfg": GoalkeeperBallVelCfg(
        speed_min=gv.speed_min,
        speed_max=gv.speed_max,
        yaw_spread_deg=yaw_spread_deg,
        pitch_min_deg=gv.pitch_min_deg,
        pitch_max_deg=gv.pitch_max_deg,
      ),
      "ball_cfg": SceneEntityCfg("ball"),
    },
  )

  _setup_robot_env(cfg)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg

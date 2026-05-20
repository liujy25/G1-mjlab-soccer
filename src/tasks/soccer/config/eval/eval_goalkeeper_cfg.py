"""Goalkeeper evaluation config — matches Humanoid-Goalkeeper paper observation space.

Adds ball position and velocity observations, plus domain randomization
events, on top of the base naive goalkeeper environment.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.soccer.config.g1.env_cfgs import unitree_g1_goalkeeper_env_cfg
from src.tasks.soccer.mdp import (
  ball_pos_in_robot_frame,
  ball_vel_in_robot_frame,
  perturb_ball_velocity,
  push_robot_base,
)

_BALL_CFG = SceneEntityCfg("ball")


def eval_goalkeeper_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Goalkeeper evaluation config matching Humanoid-Goalkeeper paper.

  Adds to the base goalkeeper config:
  - end_target_local: ball position in robot pelvis frame (actor + critic)
  - ball_vel_local: ball velocity in robot pelvis frame (critic only)
  - push_robot: interval domain randomization (15s)
  - perturb_ball_vel: interval ball velocity perturbation (0.5s)
  - randomize_joint_defaults: startup joint position randomization

  Observation noise is disabled (matching paper's eval protocol:
  add_noise=False in play mode). Domain rand events remain active.
  """
  cfg = unitree_g1_goalkeeper_env_cfg(play=play)

  # Disable observation noise for clean eval (matching paper).
  cfg.observations["actor"].enable_corruption = False

  # Add ball position to actor (no noise).
  actor_terms = dict(cfg.observations["actor"].terms)
  actor_terms["end_target_local"] = ObservationTermCfg(
    func=ball_pos_in_robot_frame,
    params={"ball_cfg": _BALL_CFG},
  )
  cfg.observations["actor"].terms = actor_terms

  # Add ball position + velocity to critic.
  critic_terms = dict(cfg.observations["critic"].terms)
  critic_terms["end_target_local"] = ObservationTermCfg(
    func=ball_pos_in_robot_frame,
    params={"ball_cfg": _BALL_CFG},
  )
  critic_terms["ball_vel_local"] = ObservationTermCfg(
    func=ball_vel_in_robot_frame,
    params={"ball_cfg": _BALL_CFG},
  )
  cfg.observations["critic"].terms = critic_terms

  ##
  # Domain randomization (matching Humanoid-GK paper)
  ##

  # Push robot every 15s, max velocity 1.5 m/s (xy), 0.5 m/s (z).
  cfg.events["push_robot"] = EventTermCfg(
    func=push_robot_base,
    mode="interval",
    interval_range_s=(15.0, 15.0),
    params={
      "vel_xy_range": (-1.5, 1.5),
      "vel_z_range": (-0.5, 0.5),
      "ang_vel_range": (0.0, 0.0),
    },
  )

  # Perturb ball velocity every 0.5s, max perturbation 0.5 m/s per axis.
  cfg.events["perturb_ball_vel"] = EventTermCfg(
    func=perturb_ball_velocity,
    mode="interval",
    interval_range_s=(0.5, 0.5),
    params={
      "vel_range": (-0.5, 0.5),
    },
  )

  # Randomize joint default positions on every reset.
  # GK paper: initial_joint_pos_scale = [0.5, 1.5], offset = [-0.1, 0.1].
  cfg.events["reset_robot_joints"] = EventTermCfg(
    func=cfg.events["reset_robot_joints"].func,
    mode="reset",
    params={
      "position_range": (-0.1, 0.1),
      "velocity_range": (-0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  )

  return cfg

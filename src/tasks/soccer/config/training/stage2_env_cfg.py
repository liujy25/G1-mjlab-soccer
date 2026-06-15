"""Stage II training config: perception-guided kicking on flat ground.

Builds on Stage I by adding ball/goal perception observations and
lightweight soccer kick rewards. The robot learns to approach and kick
balls at randomized positions while maintaining motion style.

Matches HumanoidSoccer's G1FlatKickEnvCfg.
"""

from __future__ import annotations

import copy
import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from src.tasks.soccer.config.training.stage1_env_cfg import make_stage1_env_cfg

from src.tasks.soccer.mdp.shooter_obs import (
  constant_target_point_pos,
  target_destination_pos_local,
)

from src.tasks.soccer.mdp.shooter_rewards import (
  action_rate_l2_clip,
  ball_speed_reward,
  ball_velocity_direction_alignment,
  ball_z_speed_penalty_reward,
  foot_distance,
  pelvis_orientation,
  sideways_kick,
  target_point_contact,
  target_point_proximity,
  waist_action_rate_l2_clip,
)

from src.tasks.soccer.mdp.shooter_commands import MultiMotionSoccerCommandCfg


def make_stage2_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create Stage II (perception-guided kicking) soccer environment config.

  Key changes from Stage I:
  - Uniform motion sampling (always starts from frame 0)
  - Ball position randomization (arc offset from motion trajectory)
  - Soccer observations: ball + goal in robot pelvis frame
  - Anchor position tracking weight → 0.0 (allows ball pursuit)
  - Body tracking filtered to exclude ankles (weights stay at 1.0, matching reference)
  - Soccer kick rewards: proximity, contact, sideways kick, ball vel/speed
  - Stabilization rewards: foot distance, pelvis orientation, waist smoothness
  - Ball initial velocity disabled (stationary penalty kick)
  """

  cfg = make_stage1_env_cfg()

  # -- Commands: override to uniform sampling + ball randomization -----------

  base_cmd = cfg.commands["motion"]
  cfg.commands["motion"] = MultiMotionSoccerCommandCfg(
    motion_dir="",
    motion_glob="soccer-standard-*.npz",
    anchor_body_name=base_cmd.anchor_body_name,
    body_names=base_cmd.body_names,
    entity_name="robot",
    ball_entity_name="ball",
    resampling_time_range=(1e9, 1e9),
    debug_vis=False,
    pose_range=base_cmd.pose_range,
    velocity_range=base_cmd.velocity_range,
    joint_position_range=base_cmd.joint_position_range,
    sampling_mode="uniform",
    curve_offset_range={
      "radius": (-0.25, 0.25),
      "arc_angle": math.pi / 9,
      "height": 0.11,
    },
    destination_center=(0.0, -5.0, 0.11),
    destination_length=1.0,
    destination_width=0.5,
    enable_soccer_ball_init_vel=False,
  )

  # -- Observations: add soccer perception to actor and critic ----------------

  cfg.observations["actor"].terms["target_point_pos"] = ObservationTermCfg(
    func=constant_target_point_pos,
    params={"command_name": "motion"},
  )
  cfg.observations["actor"].terms["target_destination_pos"] = ObservationTermCfg(
    func=target_destination_pos_local,
    params={"command_name": "motion"},
  )

  cfg.observations["critic"].terms["target_point_pos"] = ObservationTermCfg(
    func=constant_target_point_pos,
    params={"command_name": "motion"},
  )
  cfg.observations["critic"].terms["target_destination_pos"] = ObservationTermCfg(
    func=target_destination_pos_local,
    params={"command_name": "motion"},
  )

  # -- Rewards: match reference G1FlatProximityEnvCfg weights ------------------

  # Disable global anchor position tracking (allows robot to pursue ball).
  cfg.rewards["track_anchor_pos"] = RewardTermCfg(
    func=cfg.rewards["track_anchor_pos"].func,
    weight=0.0,
    params={"command_name": "motion", "std": 0.3},
  )
  cfg.rewards["track_anchor_ori"] = RewardTermCfg(
    func=cfg.rewards["track_anchor_ori"].func,
    weight=0.5,
    params={"command_name": "motion", "std": 0.4},
  )

  # Use filtered body tracking (exclude ankles for positional generalization,
  # matching reference G1FlatProximityEnvCfg body_names subset).
  cfg.rewards["track_body_pos"] = RewardTermCfg(
    func=cfg.rewards["track_body_pos"].func,
    weight=0.8,
    params={
      "command_name": "motion",
      "std": 0.3,
      "body_names": (
        "pelvis",
        "left_hip_roll_link", "left_knee_link",
        "right_hip_roll_link", "right_knee_link",
        "torso_link",
        "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
        "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
      ),
    },
  )
  cfg.rewards["track_body_ori"] = RewardTermCfg(
    func=cfg.rewards["track_body_ori"].func,
    weight=0.8,
    params={
      "command_name": "motion",
      "std": 0.4,
      "body_names": (
        "pelvis",
        "left_hip_roll_link", "left_knee_link",
        "right_hip_roll_link", "right_knee_link",
        "torso_link",
        "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
        "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
      ),
    },
  )
  cfg.rewards["track_body_lin_vel"] = RewardTermCfg(
    func=cfg.rewards["track_body_lin_vel"].func,
    weight=0.8,
    params={"command_name": "motion", "std": 1.0},
  )
  cfg.rewards["track_body_ang_vel"] = RewardTermCfg(
    func=cfg.rewards["track_body_ang_vel"].func,
    weight=0.8,
    params={"command_name": "motion", "std": 3.14},
  )

  # Soccer rewards.
  _foot_names = ("left_ankle_roll_link", "right_ankle_roll_link")
  _waist_joints = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")

  cfg.rewards["proximity"] = RewardTermCfg(
    func=target_point_proximity,
    weight=1.0,
    params={"std": 4.0, "command_name": "motion"},
  )
  cfg.rewards["contact"] = RewardTermCfg(
    func=target_point_contact,
    weight=50.0,
    params={
      "command_name": "motion",
      "ball_sensor_name": "ball_robot_contact",
      "horizontal_force_threshold": 10.0,
      "foot_body_names": _foot_names,
    },
  )
  cfg.rewards["sideways_kick"] = RewardTermCfg(
    func=sideways_kick,
    weight=50.0,
    params={
      "command_name": "motion",
      "ball_sensor_name": "ball_robot_contact",
      "horizontal_force_threshold": 10.0,
      "foot_body_names": _foot_names,
    },
  )
  cfg.rewards["ball_vel_align"] = RewardTermCfg(
    func=ball_velocity_direction_alignment,
    weight=30.0,
    params={
      "command_name": "motion",
      "std": 0.8,
      "velocity_threshold": 0.5,
      "ball_sensor_name": "ball_robot_contact",
      "horizontal_force_threshold": 10.0,
      "foot_body_names": _foot_names,
    },
  )
  cfg.rewards["ball_speed"] = RewardTermCfg(
    func=ball_speed_reward,
    weight=10.0,
    params={
      "command_name": "motion",
      "std": 1.2,
      "velocity_threshold": 0.5,
      "ball_sensor_name": "ball_robot_contact",
      "horizontal_force_threshold": 10.0,
      "foot_body_names": _foot_names,
    },
  )
  cfg.rewards["ball_z_speed"] = RewardTermCfg(
    func=ball_z_speed_penalty_reward,
    weight=-0.2,
    params={
      "command_name": "motion",
      "std": 3.0,
      "velocity_threshold": 0.5,
      "ball_sensor_name": "ball_robot_contact",
      "horizontal_force_threshold": 10.0,
      "foot_body_names": _foot_names,
    },
  )

  # Stabilization rewards.
  cfg.rewards["foot_distance"] = RewardTermCfg(
    func=foot_distance,
    weight=0.2,
    params={
      "threshold": 0.24,
      "std": 0.5,
      "entity_name": "robot",
      "body_names": _foot_names,
    },
  )
  cfg.rewards["pelvis_orientation"] = RewardTermCfg(
    func=pelvis_orientation,
    weight=-1.0,
    params={"command_name": "motion"},
  )
  cfg.rewards["waist_action_rate"] = RewardTermCfg(
    func=waist_action_rate_l2_clip,
    weight=-0.25,
    params={
      "entity_name": "robot",
      "joint_names": _waist_joints,
    },
  )

  # Foot position tracking — matches reference G1FlatProximityEnvCfg.
  cfg.rewards["track_foot_pos"] = RewardTermCfg(
    func=cfg.rewards["track_body_pos"].func,
    weight=1.0,
    params={
      "command_name": "motion",
      "std": 0.3,
      "body_names": _foot_names,
    },
  )

  return cfg

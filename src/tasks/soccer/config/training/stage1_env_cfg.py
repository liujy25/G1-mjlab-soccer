"""Stage I training config: motion tracking on flat ground with adaptive sampling.

The robot learns to imitate diverse kick motion references without
perceptual inputs. This establishes stable motion priors before Stage II
adds ball perception and kick rewards.

Matches HumanoidSoccer's G1TerrainMotionEnvCfg (flat variant).
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from src.tasks.soccer import mdp
from src.tasks.soccer.ball import get_ball_cfg
from src.tasks.soccer.ground import get_ground_cfg
from src.tasks.soccer.config.soccer_settings import SETTINGS
from src.tasks.soccer.soccer_env_cfg import _add_soccer_scene_postproc


# Reuse mjlab's built-in motion tracking reward functions.
from mjlab.tasks.tracking.mdp.rewards import (  # type: ignore[import-untyped]
  motion_global_anchor_position_error_exp,
  motion_global_anchor_orientation_error_exp,
  motion_relative_body_position_error_exp,
  motion_relative_body_orientation_error_exp,
  motion_global_body_linear_velocity_error_exp,
  motion_global_body_angular_velocity_error_exp,
)

# Soccer training observation functions (MultiMotionSoccerCommand-aware).
from src.tasks.soccer.mdp.shooter_obs import (
  motion_anchor_ang_vel as _motion_anchor_ang_vel,
  robot_body_pos_b,
  robot_body_ori_b,
  constant_target_point_pos,
  target_destination_pos_local,
)

# Soccer reward functions.
from src.tasks.soccer.mdp.shooter_rewards import (
  action_rate_l2_clip,
  foot_distance,
  undesired_contacts,
)

# Soccer-specific command.
from src.tasks.soccer.mdp.shooter_commands import MultiMotionSoccerCommandCfg

# mjlab built-in.
from mjlab.envs.mdp.rewards import joint_pos_limits


# Body names for motion tracking (14 bodies matching the reference).
TRACKING_BODY_NAMES = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)


def make_stage1_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create Stage I (motion tracking) soccer environment config.

  Key design choices:
  - Adaptive motion sampling: focuses training on failure-prone phases
  - Asymmetric actor-critic: critic sees privileged body poses
  - Pure tracking rewards: no ball perception or kick rewards
  - Motion-reference terminations: safety guardrails for training
  """

  # -- Commands ---------------------------------------------------------------

  commands = {
    "motion": MultiMotionSoccerCommandCfg(
      motion_dir="",
      motion_glob="*.npz",
      anchor_body_name="torso_link",
      body_names=TRACKING_BODY_NAMES,
      entity_name="robot",
      ball_entity_name="ball",
      resampling_time_range=(1e9, 1e9),
      debug_vis=False,
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      velocity_range={
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.2, 0.2),
        "roll": (-0.52, 0.52),
        "pitch": (-0.52, 0.52),
        "yaw": (-0.78, 0.78),
      },
      joint_position_range=(-0.1, 0.1),
      sampling_mode="adaptive",
      adaptive_kernel_size=3,
      adaptive_lambda=0.1,
      adaptive_uniform_ratio=0.1,
      adaptive_alpha=0.4,
    ),
  }

  # -- Observations -----------------------------------------------------------

  actor_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "motion"},
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "motion_ref_ang_vel": ObservationTermCfg(
      func=_motion_anchor_ang_vel,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "target_point_pos": ObservationTermCfg(
      func=constant_target_point_pos,
      params={"command_name": "motion"},
    ),
    "target_destination_pos": ObservationTermCfg(
      func=target_destination_pos_local,
      params={"command_name": "motion"},
    ),
  }

  critic_terms = {
    **actor_terms,
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.shooter_obs.motion_anchor_pos_b,
      params={"command_name": "motion"},
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.shooter_obs.motion_anchor_ori_b,
      params={"command_name": "motion"},
    ),
    "body_pos": ObservationTermCfg(
      func=robot_body_pos_b,
      params={"command_name": "motion"},
    ),
    "body_ori": ObservationTermCfg(
      func=robot_body_ori_b,
      params={"command_name": "motion"},
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=1,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
    ),
  }

  # -- Actions ----------------------------------------------------------------

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,
      use_default_offset=True,
    )
  }

  # -- Events -----------------------------------------------------------------

  events = {
    # Robot and ball state are set by MultiMotionSoccerCommand._resample_command.
    # No reset events needed — they would overwrite the command's state.
    "push_robot": EventTermCfg(
      func=mdp.push_robot_base,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "vel_xy_range": (-0.5, 0.5),
        "vel_z_range": (-0.2, 0.2),
        "ang_vel_range": (-0.52, 0.52),
      },
    ),
  }

  # -- Rewards ----------------------------------------------------------------

  rewards = {
    # Motion tracking.
    "track_anchor_pos": RewardTermCfg(
      func=motion_global_anchor_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    "track_anchor_ori": RewardTermCfg(
      func=motion_global_anchor_orientation_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    "track_body_pos": RewardTermCfg(
      func=motion_relative_body_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    "track_body_ori": RewardTermCfg(
      func=motion_relative_body_orientation_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    "track_body_lin_vel": RewardTermCfg(
      func=motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0},
    ),
    "track_body_ang_vel": RewardTermCfg(
      func=motion_global_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 3.14},
    ),
    # Regularization.
    "action_rate": RewardTermCfg(func=action_rate_l2_clip, weight=-0.1),
    "joint_limit": RewardTermCfg(
      func=joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "undesired_contact": RewardTermCfg(
      func=undesired_contacts,
      weight=-0.1,
      params={
        "sensor_name": "contact_forces",
        "threshold": 1.0,
        "excluded_body_names": (
          "left_ankle_roll_link",
          "right_ankle_roll_link",
          "left_wrist_yaw_link",
          "right_wrist_yaw_link",
        ),
      },
    ),
  }

  # -- Terminations -----------------------------------------------------------

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    "anchor_pos_z": TerminationTermCfg(
      func=mdp.bad_anchor_pos_z,
      params={"command_name": "motion", "threshold": 0.25},
    ),
    "anchor_ori": TerminationTermCfg(
      func=mdp.bad_anchor_ori,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "threshold": 0.8,
        "command_name": "motion",
      },
    ),
    "ee_body_pos": TerminationTermCfg(
      func=mdp.bad_ee_body_pos_z,
      params={
        "command_name": "motion",
        "threshold": 0.25,
        "body_names": (
          "left_ankle_roll_link", "right_ankle_roll_link",
          "left_wrist_yaw_link", "right_wrist_yaw_link",
        ),
      },
    ),
  }

  # -- Assemble ---------------------------------------------------------------

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      entities={
        "ground": get_ground_cfg(),
        "ball": get_ball_cfg(),
      },
      num_envs=1,
      env_spacing=2.5,
      spec_fn=_add_soccer_scene_postproc,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=6.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=48,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=SETTINGS.episode_length_s,
  )

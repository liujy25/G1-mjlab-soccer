"""Shooter evaluation config — matches HumanoidSoccer paper observation space.

Observation space (matching paper Eq.1, Section III-A):

  o_t = (o^prop_t, o^ref_t, o^soc_t)

  o^prop_t (93D proprioception):
    - projected_gravity (3D)
    - base_ang_vel      (3D)
    - joint_pos_rel     (29D)
    - joint_vel_rel     (29D)
    - last_action       (29D)

  o^ref_t (61D motion tracking reference):
    - motion_ref_joint_pos      (29D) — reference joint positions at current frame
    - motion_ref_joint_vel      (29D) — reference joint velocities at current frame
    - motion_ref_anchor_ang_vel (3D)  — reference anchor angular velocity

  o^soc_t (6D soccer perception):
    - target_point_pos       (3D) — ball position in robot pelvis frame
    - target_destination_pos (3D) — goal center in robot pelvis frame

  Total actor:  93 + 61 + 6 = 160D
  Total critic: 96 + 61 + 6 = 163D (adds base_lin_vel)

Domain randomization (eval only — no sim2real):
  - Robot root pos: ±0.5m xy circle behind the ball (reset)
  - Robot root yaw: ±0.2 rad (reset)
  - Joint default pos: ±0.01 rad (reset)

Observation noise: disabled (paper eval protocol).
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from src.tasks.soccer.config.g1.env_cfgs import unitree_g1_shooter_env_cfg
from src.tasks.soccer.config.soccer_settings import SETTINGS
from src.tasks.soccer import mdp as soccer_mdp

# Goal center world-frame position (center of goal opening).
_GOAL_CENTER = (0.0, 0.0, SETTINGS.goal.height / 2)  # (0, 0, 0.9)
_BALL_CFG = SceneEntityCfg("ball")
_ROBOT_JOINT_CFG = SceneEntityCfg("robot", joint_names=(".*",))


def eval_shooter_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Shooter evaluation config matching HumanoidSoccer paper.

  Observation space:
    actor  = proprioception(93) + motion_ref(61) + soccer(6) = 160D
    critic = actor(160) + base_lin_vel(3) = 163D

  Domain randomization (eval only):
    - Robot root pos: random within 0.5m radius circle behind ball
    - Robot root yaw: ±0.2 rad
    - Joint default pos: ±0.01 rad

  Ball position is fixed (per settings.yaml).
  """
  cfg = unitree_g1_shooter_env_cfg(play=play)

  ##
  # Rebuild actor observations — match paper.
  ##

  # O^prop_t: proprioception (93D).
  actor_prop_terms = {
    "projected_gravity": ObservationTermCfg(func=soccer_mdp.projected_gravity),
    "base_ang_vel": ObservationTermCfg(
      func=soccer_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
    ),
    "joint_pos": ObservationTermCfg(func=soccer_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=soccer_mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=soccer_mdp.last_action),
  }

  # O^ref_t: motion tracking reference (61D).
  actor_ref_terms = {
    "motion_ref_joint_pos": ObservationTermCfg(func=soccer_mdp.motion_ref_joint_pos),
    "motion_ref_joint_vel": ObservationTermCfg(func=soccer_mdp.motion_ref_joint_vel),
    "motion_ref_anchor_ang_vel": ObservationTermCfg(func=soccer_mdp.motion_ref_anchor_ang_vel),
  }

  # O^soc_t: soccer perception (6D).
  actor_soc_terms = {
    "target_point_pos": ObservationTermCfg(
      func=soccer_mdp.ball_pos_in_robot_frame,
      params={"ball_cfg": _BALL_CFG},
    ),
    "target_destination_pos": ObservationTermCfg(
      func=soccer_mdp.world_point_in_robot_frame,
      params={"point": _GOAL_CENTER},
    ),
  }

  all_actor_terms = {**actor_prop_terms, **actor_ref_terms, **actor_soc_terms}

  cfg.observations["actor"] = ObservationGroupCfg(
    terms=all_actor_terms,
    concatenate_terms=True,
    enable_corruption=False,  # paper eval protocol: no observation noise
    history_length=1,
  )

  # O critic = actor + base_lin_vel.
  all_critic_terms = {
    **all_actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=soccer_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
  }
  cfg.observations["critic"] = ObservationGroupCfg(
    terms=all_critic_terms,
    concatenate_terms=True,
    enable_corruption=False,
    history_length=1,
  )

  ##
  # Domain randomization (eval only — no sim2real physics DR).
  ##

  # Robot root pos: random behind ball.  x only backward (ball is at +x from robot),
  # so robot stays at least shooter_behind_ball (1.0m) from the ball surface.
  cfg.events["reset_robot_base"] = EventTermCfg(
    func=soccer_mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {
        "x": (-0.5, 0.0),   # only backward (ball at +x direction)
        "y": (-0.5, 0.5),   # lateral
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (-0.5, 0.5), # ±0.5 rad
      },
      "velocity_range": {},
    },
  )

  # Randomize joint default positions on every reset (±0.01 rad).
  cfg.events["reset_robot_joints"] = EventTermCfg(
    func=soccer_mdp.reset_joints_by_offset,
    mode="reset",
    params={
      "position_range": (-0.01, 0.01),
      "velocity_range": (-0.0, 0.0),
      "asset_cfg": _ROBOT_JOINT_CFG,
    },
  )

  # Ball reset: fixed position (no velocity), per settings.yaml.
  cfg.events["reset_ball"] = EventTermCfg(
    func=soccer_mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {},
      "velocity_range": {},
      "asset_cfg": _BALL_CFG,
    },
  )

  # Remove push_robot interval DR (sim2real only).
  cfg.events.pop("push_robot", None)

  ##
  # Terminations — paper's motion-tracking safety constraints.
  # Disabled by default (too strict for zero-agent baseline).
  # Uncomment to enable when training a tracking policy.
  ##

  # cfg.terminations["anchor_pos_z"] = TerminationTermCfg(
  #   func=soccer_mdp.bad_anchor_pos_z,
  #   params={"threshold": 0.25},
  # )
  # cfg.terminations["anchor_ori"] = TerminationTermCfg(
  #   func=soccer_mdp.bad_anchor_ori,
  #   params={"threshold": 0.8},
  # )
  # cfg.terminations["ee_body_pos"] = TerminationTermCfg(
  #   func=soccer_mdp.bad_ee_body_pos_z,
  #   params={"threshold": 0.25},
  # )

  return cfg

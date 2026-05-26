"""Cross-evaluate two teams: Shooter vs Goalkeeper in a shared MuJoCo scene.

compete.py reads raw MuJoCo state (joint positions, root poses, ball pos/vel)
and sends it to each team's policy server via REST API.  Teams compute their
own observations from raw state, so observation spaces are fully decoupled.

Scene layout (world frame, z-up):
  - Goalkeeper at (0, 0, 0.8), yaw=0, faces +x
  - Goal at (-0.5, 0, 0), behind the goalkeeper
  - Shooter at (4, 0, 0.8), yaw=pi, faces -x toward the goal
  - Ball at (3, 0, 0.1), in front of the shooter

Usage:
  # Headless batch (10 trials, Phase 2 default)
  python scripts/compete.py \
      --shooter-api http://<team_a_ip>:8000 \
      --goalkeeper-api http://<team_b_ip>:8001 \
      --headless --num-trials 10

  # Interactive viewer (single episode, for debugging)
  python scripts/compete.py \
      --shooter-api http://<team_a_ip>:8000 \
      --goalkeeper-api http://<team_b_ip>:8001

  # Zero-agent baseline (no policy servers)
  python scripts/compete.py --headless --num-trials 10

-------------------------------------------------------------------------------
RAW STATE PROTOCOL
-------------------------------------------------------------------------------

compete.py sends the same raw state dict to both policy servers:

  {
    "shooter": {
      "root_pos": [x,y,z], "root_quat": [w,x,y,z],
      "root_lin_vel": [vx,vy,vz], "root_ang_vel": [wx,wy,wz],
      "joint_pos": [...29], "joint_vel": [...29],
      "last_action": [...29]
    },
    "goalkeeper": { "... same structure ..." },
    "ball": { "pos": [x,y,z], "vel": [vx,vy,vz] }
  }

Each server responds with  {"action": [[...29]]}.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import requests
import torch
import tyro

# ---------------------------------------------------------------------------
# mjlab / project imports
# ---------------------------------------------------------------------------

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer, ViewerConfig

from src.assets.robots import get_g1_robot_cfg, G1_ACTION_SCALE
from src.assets.robots.unitree_g1.g1_constants import HOME_KEYFRAME, FULL_COLLISION
from src.tasks.soccer.ball import get_ball_cfg
from src.tasks.soccer.goal import get_goal_cfg
from src.tasks.soccer.ground import get_ground_cfg
from src.tasks.soccer.soccer_env_cfg import _add_soccer_scene_postproc
from src.tasks.soccer import mdp
from src.tasks.soccer.mdp.goalkeeper_obs import _GK_DEFAULT_JOINT_POS, get_gk_robot_cfg

# =============================================================================
# Scene constants
# =============================================================================

GK_POS: tuple[float, float, float] = (0.0, 0.0, 0.8)
GK_YAW: float = 0.0
GOAL_POS: tuple[float, float, float] = (-0.5, 0.0, 0.0)
SHOOTER_POS: tuple[float, float, float] = (4.0, 0.0, 0.8)
SHOOTER_YAW: float = math.pi
BALL_POS: tuple[float, float, float] = (3.0, 0.0, 0.1)
EPISODE_LENGTH_S: float = 10.0
CONTROL_DECIMATION: int = 4
MUJOCO_TIMESTEP: float = 0.005
GOAL_X: float = -0.5
GOAL_HALF_WIDTH: float = 1.5
GOAL_HEIGHT: float = 1.8

_SHOOTER_CFG = SceneEntityCfg("shooter")
_GK_CFG = SceneEntityCfg("goalkeeper")
_BALL_CFG = SceneEntityCfg("ball")


# =============================================================================
# Helper: yaw -> quaternion  (rotation about world z-axis)
# =============================================================================

def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half = yaw / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


# =============================================================================
# Raw state extraction
# =============================================================================

def _build_raw_state(
    env_base: ManagerBasedRlEnv,
    prev_action_shooter: torch.Tensor,
    prev_action_gk: torch.Tensor,
) -> dict:
    """Read raw MuJoCo state for both robots and the ball.

    prev_action_* tensors have shape (1, 29).
    """
    scene = env_base.scene
    shooter = scene["shooter"]
    goalkeeper = scene["goalkeeper"]
    ball = scene["ball"]

    def _robot_state(robot) -> dict:
        return {
            "root_pos": robot.data.root_link_pos_w[0].cpu().tolist(),
            "root_quat": robot.data.root_link_quat_w[0].cpu().tolist(),
            "root_lin_vel": robot.data.root_link_lin_vel_w[0].cpu().tolist(),
            "root_ang_vel": robot.data.root_link_ang_vel_w[0].cpu().tolist(),
            "joint_pos": robot.data.joint_pos[0].cpu().tolist(),
            "joint_vel": robot.data.joint_vel[0].cpu().tolist(),
        }

    return {
        "shooter": {
            **_robot_state(shooter),
            "last_action": prev_action_shooter[0].cpu().tolist(),
        },
        "goalkeeper": {
            **_robot_state(goalkeeper),
            "last_action": prev_action_gk[0].cpu().tolist(),
        },
        "ball": {
            "pos": ball.data.root_link_pos_w[0].cpu().tolist(),
            "vel": ball.data.root_link_vel_w[0, :3].cpu().tolist(),
        },
    }


# =============================================================================
# Robot entity factories
# =============================================================================

def _make_shooter_robot() -> Any:
    """Standard G1 at SHOOTER_POS, yaw=pi (faces -x toward goalkeeper)."""
    cfg = get_g1_robot_cfg()
    cfg.init_state = replace(
        HOME_KEYFRAME,
        pos=SHOOTER_POS,
        rot=_yaw_to_quat(SHOOTER_YAW),
    )
    cfg.collisions = (FULL_COLLISION,)
    return cfg


def _make_goalkeeper_robot() -> Any:
    """G1 with GK-specific PD gains and GK reference stance at GK_POS."""
    cfg = get_gk_robot_cfg()
    cfg.init_state = replace(
        cfg.init_state,
        pos=GK_POS,
        rot=_yaw_to_quat(GK_YAW),
        joint_pos=_GK_DEFAULT_JOINT_POS,
    )
    cfg.collisions = (FULL_COLLISION,)
    return cfg


# =============================================================================
# Compete Environment Config  (physics only — no observation groups needed)
# =============================================================================

def make_compete_env_cfg() -> ManagerBasedRlEnvCfg:
    """Build the two-robot competition environment configuration.

    No observation groups are defined — compete.py reads raw MuJoCo state
    directly and sends it to the remote policy servers.
    """

    entities: dict[str, Any] = {
        "ground": get_ground_cfg(),
        "ball": get_ball_cfg(pos=BALL_POS),
        "goal": get_goal_cfg(pos=GOAL_POS),
        "shooter": _make_shooter_robot(),
        "goalkeeper": _make_goalkeeper_robot(),
    }

    # -- Actions --------------------------------------------------------------

    actions: dict[str, ActionTermCfg] = {
        "shooter_joint_pos": JointPositionActionCfg(
            entity_name="shooter",
            actuator_names=(".*",),
            scale=G1_ACTION_SCALE,
            use_default_offset=True,
        ),
        "goalkeeper_joint_pos": JointPositionActionCfg(
            entity_name="goalkeeper",
            actuator_names=(".*",),
            scale=0.25,
            use_default_offset=True,
        ),
    }

    # -- Events (reset) -------------------------------------------------------

    events: dict[str, EventTermCfg] = {
        "reset_shooter_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": _SHOOTER_CFG,
            },
        ),
        "reset_shooter_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.0, 0.0),
                "velocity_range": (-0.0, 0.0),
                "asset_cfg": SceneEntityCfg("shooter", joint_names=(".*",)),
            },
        ),
        "reset_goalkeeper_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": _GK_CFG,
            },
        ),
        "reset_goalkeeper_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.0, 0.0),
                "velocity_range": (-0.0, 0.0),
                "asset_cfg": SceneEntityCfg("goalkeeper", joint_names=(".*",)),
            },
        ),
        "reset_ball": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": _BALL_CFG,
            },
        ),
    }

    # -- Terminations ---------------------------------------------------------

    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "shooter_fell": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"asset_cfg": _SHOOTER_CFG, "limit_angle": math.radians(70.0)},
        ),
        "goalkeeper_fell": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"asset_cfg": _GK_CFG, "limit_angle": math.radians(70.0)},
        ),
    }

    # -- Rewards (placeholder — unused at inference) --------------------------

    rewards: dict[str, RewardTermCfg] = {
        "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    }

    # -- Assemble & return ----------------------------------------------------

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities=entities,
            num_envs=1,
            spec_fn=_add_soccer_scene_postproc,
        ),
        observations={
            # Dummy terms — mjlab requires at least one term per group for
            # initialization. Raw state is read directly from MuJoCo instead.
            "shooter_actor": ObservationGroupCfg(
                terms={
                    "dummy": ObservationTermCfg(
                        func=mdp.builtin_sensor,
                        params={"sensor_name": "shooter/imu_ang_vel"},
                    ),
                },
                concatenate_terms=True, enable_corruption=False,
                history_length=1,
            ),
            "goalkeeper_actor": ObservationGroupCfg(
                terms={
                    "dummy": ObservationTermCfg(
                        func=mdp.builtin_sensor,
                        params={"sensor_name": "goalkeeper/imu_ang_vel"},
                    ),
                },
                concatenate_terms=True, enable_corruption=False,
                history_length=1,
            ),
        },
        actions=actions,
        commands={},
        events=events,
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            lookat=(2.0, 0.0, 1.0),
            distance=6.0,
            elevation=-15.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=256,
            njmax=3000,
            mujoco=MujocoCfg(
                timestep=MUJOCO_TIMESTEP,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=CONTROL_DECIMATION,
        episode_length_s=EPISODE_LENGTH_S,
    )


# =============================================================================
# Zero-policy fallback
# =============================================================================

class _ZeroPolicy:
    """Policy that always outputs zeros (baseline / debug)."""

    def __init__(self, action_dim: int, device: str):
        self._zero = torch.zeros(1, action_dim, device=device)

    def __call__(self, _input: Any) -> torch.Tensor:
        return self._zero

    def reset(self) -> None:
        pass


# =============================================================================
# REST API policy client
# =============================================================================

class ApiPolicy:
    """Policy that delegates to a remote server via the raw state protocol.

    Parameters
    ----------
    url : str
        Base URL of the policy server (e.g. ``http://team-a:8000``).
    action_dim : int
        Expected action dimension.
    device : str
        Torch device for the output tensor.
    timeout : float
        HTTP request timeout in seconds.
    """

    def __init__(
        self, url: str, action_dim: int, device: str, timeout: float = 2.0,
    ):
        self._url = url.rstrip("/")
        self._device = device
        self._timeout = timeout
        try:
            resp = requests.post(
                f"{self._url}/reset", json={}, timeout=self._timeout,
            )
            resp.raise_for_status()
            print(f"[INFO] API policy connected: {self._url}  (act_dim={action_dim})")
        except requests.RequestException as e:
            print(f"[WARN] API policy at {self._url} is not reachable: {e}")
            print(f"       Make sure the server is running before starting trials.")

    def __call__(self, raw_state: dict) -> torch.Tensor:
        """Send raw MuJoCo state to remote server and return action tensor."""
        try:
            resp = requests.post(
                f"{self._url}/act",
                json=raw_state,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            action = torch.tensor(
                resp.json()["action"], device=self._device, dtype=torch.float32,
            )
            return action
        except requests.RequestException as e:
            raise RuntimeError(
                f"API call to {self._url}/act failed: {e}"
            ) from e

    def reset(self) -> None:
        """Signal the remote server to reset its policy state."""
        try:
            requests.post(
                f"{self._url}/reset", json={}, timeout=self._timeout,
            )
        except requests.RequestException:
            pass  # best-effort reset


# =============================================================================
# Combined policy wrapper  (for viewer)
# =============================================================================

class CombinedPolicy:
    """Wraps two independent policies so the viewer sees a single policy."""

    def __init__(
        self,
        shooter_policy: Any,
        goalkeeper_policy: Any,
        env_base: ManagerBasedRlEnv,
        device: str,
    ):
        self._shooter = shooter_policy
        self._goalkeeper = goalkeeper_policy
        self._env_base = env_base
        self._device = device
        self._prev_action_s = torch.zeros(1, 29, device=device)
        self._prev_action_g = torch.zeros(1, 29, device=device)

    def __call__(self, _obs: dict) -> torch.Tensor:
        raw = _build_raw_state(
            self._env_base, self._prev_action_s, self._prev_action_g,
        )
        s_act = self._shooter(raw)
        g_act = self._goalkeeper(raw)
        action = torch.cat([s_act, g_act], dim=-1)
        self._prev_action_s = s_act.detach().clone()
        self._prev_action_g = g_act.detach().clone()
        return action

    def reset(self) -> None:
        self._shooter.reset()
        self._goalkeeper.reset()
        self._prev_action_s.zero_()
        self._prev_action_g.zero_()


# =============================================================================
# Competition metrics
# =============================================================================

def _ball_entered_goal(ball_pos: torch.Tensor) -> bool:
    """Ball has crossed the goal plane (x <= GOAL_X) inside the goal frame."""
    x, y, z = ball_pos[0].item(), ball_pos[1].item(), ball_pos[2].item()
    return x <= GOAL_X and abs(y) <= GOAL_HALF_WIDTH and z <= GOAL_HEIGHT


def _ball_blocked(
    ball_vel: torch.Tensor,
    ball_pos: torch.Tensor,
    prev_speed: float,
    threshold: float = 2.0,
) -> bool:
    """Ball was blocked if its speed dropped by > threshold m/s behind GK."""
    speed = float(torch.norm(ball_vel))
    behind_gk = ball_pos[0].item() <= GK_POS[0]
    return behind_gk and (prev_speed - speed) > threshold


# =============================================================================
# Trial runner
# =============================================================================

def run_trial(
    env: ManagerBasedRlEnv,
    env_base: ManagerBasedRlEnv,
    shooter_policy: Any,
    goalkeeper_policy: Any,
    max_steps: int = 500,
) -> dict[str, Any]:
    """Run one competition episode using raw state protocol.

    Returns a dict with keys:
      goal_scored, blocked, steps, ball_final_x, early_termination
    """
    env.reset()
    shooter_policy.reset()
    goalkeeper_policy.reset()

    device = env.unwrapped.device
    prev_action_s = torch.zeros(1, 29, device=device)
    prev_action_g = torch.zeros(1, 29, device=device)

    ball = env.unwrapped.scene["ball"]
    goal_scored = False
    blocked = False
    prev_ball_speed = 0.0
    steps = 0
    early_termination = False

    for _ in range(max_steps):
        with torch.inference_mode():
            raw = _build_raw_state(env_base, prev_action_s, prev_action_g)
            s_act = shooter_policy(raw)
            g_act = goalkeeper_policy(raw)

        action = torch.cat([s_act, g_act], dim=-1)
        result = env.step(action)
        steps += 1

        prev_action_s = s_act.detach().clone()
        prev_action_g = g_act.detach().clone()

        ball_pos = ball.data.root_link_pos_w[0].cpu()
        ball_vel = ball.data.root_link_vel_w[0, :3].cpu()
        ball_speed = float(torch.norm(ball_vel))

        if _ball_entered_goal(ball_pos):
            goal_scored = True

        if _ball_blocked(ball_vel, ball_pos, prev_ball_speed):
            blocked = True

        prev_ball_speed = ball_speed

        terminated = result[2]
        if hasattr(terminated, "item"):
            terminated = bool(terminated.item())
        else:
            terminated = bool(terminated)
        if terminated:
            if steps < max_steps - 1:
                early_termination = True
            break

    return {
        "goal_scored": goal_scored,
        "blocked": blocked,
        "steps": steps,
        "ball_final_x": float(ball.data.root_link_pos_w[0, 0].cpu()),
        "early_termination": early_termination,
    }


# =============================================================================
# Headless batch evaluation
# =============================================================================

def run_headless_eval(
    num_trials: int,
    env: ManagerBasedRlEnv,
    env_base: ManagerBasedRlEnv,
    shooter_policy: Any,
    goalkeeper_policy: Any,
) -> dict[str, Any]:
    """Run multiple trials headless and return aggregate statistics."""
    print(f"\n[INFO] Running {num_trials} headless competition trials ...\n")

    goals = 0
    blocks = 0
    early_terminations = 0
    ball_crossed_goal_line = 0

    for trial in range(num_trials):
        stats = run_trial(env, env_base, shooter_policy, goalkeeper_policy)
        if stats["goal_scored"]:
            goals += 1
        if stats["blocked"]:
            blocks += 1
        if stats["early_termination"]:
            early_terminations += 1
        if stats["ball_final_x"] <= GOAL_X:
            ball_crossed_goal_line += 1

        interval = 1 if num_trials <= 10 else (num_trials // 10)
        if (trial + 1) % interval == 0 or trial == 0:
            print(
                f"  Trial {trial + 1:3d}/{num_trials}: "
                f"goal={stats['goal_scored']}, "
                f"blocked={stats['blocked']}, "
                f"early_term={stats['early_termination']}, "
                f"steps={stats['steps']}"
            )

    total = num_trials
    print(f"\n{'=' * 60}")
    print(f"  Competition Summary  ({total} trials)")
    print(f"{'=' * 60}")
    print(f"  Shooter goals:          {goals}/{total}  ({goals / total * 100:.1f}%)")
    print(f"  Goalkeeper blocks:      {blocks}/{total}  ({blocks / total * 100:.1f}%)")
    print(f"  Early terminations:     {early_terminations}/{total}")
    print(f"  Ball crossed goal line: {ball_crossed_goal_line}/{total}")
    print(f"{'=' * 60}\n")

    return {
        "num_trials": total,
        "goals": goals,
        "blocks": blocks,
        "early_terminations": early_terminations,
        "ball_crossed_goal_line": ball_crossed_goal_line,
    }


# =============================================================================
# Viewer
# =============================================================================

def run_viewer(
    viewer_type: str,
    env: ManagerBasedRlEnv,
    combined_policy: CombinedPolicy,
) -> None:
    """Launch an interactive viewer with the combined policy."""
    if viewer_type == "native":
        NativeMujocoViewer(env, combined_policy).run()
    elif viewer_type == "viser":
        ViserPlayViewer(env, combined_policy).run()
    else:
        raise RuntimeError(f"Unsupported viewer: {viewer_type}")


# =============================================================================
# CLI
# =============================================================================

@dataclass
class CompeteConfig:
    shooter_api: str | None = None
    """REST API URL for the shooter policy (Phase 2 tournament)."""

    goalkeeper_api: str | None = None
    """REST API URL for the goalkeeper policy (Phase 2 tournament)."""

    num_trials: int = 0
    """Number of evaluation trials (> 0 enables headless batch mode)."""

    headless: bool = False
    """Run without a viewer (required for multi-trial eval)."""

    video: bool = False
    """Record video of the first trial (requires --headless)."""

    video_length: int = 500
    """Video length in steps."""

    video_height: int = 480
    video_width: int = 640

    viewer: str = "auto"
    """Viewer type: 'auto', 'native', or 'viser'."""

    device: str | None = None
    """Torch device (auto-detected if omitted)."""

    seed: int = 2810
    """Random seed."""

    task_id: str = "Compete"


def run_compete(cfg: CompeteConfig) -> None:
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # -- Build environment ----------------------------------------------------
    print(f"Task: {cfg.task_id}  |  device: {device}")
    env_cfg = make_compete_env_cfg()
    env_cfg.scene.num_envs = 1
    env_cfg.viewer.height = cfg.video_height
    env_cfg.viewer.width = cfg.video_width

    render_mode = "rgb_array" if cfg.video else None
    env_base = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
    act_dim_shooter = env_base.action_manager.get_term("shooter_joint_pos").action_dim
    act_dim_goalkeeper = env_base.action_manager.get_term("goalkeeper_joint_pos").action_dim
    print(f"Action dims: shooter={act_dim_shooter}, goalkeeper={act_dim_goalkeeper}")

    # -- Load policies --------------------------------------------------------
    if cfg.shooter_api:
        shooter_policy = ApiPolicy(cfg.shooter_api, act_dim_shooter, device)
    else:
        shooter_policy = _ZeroPolicy(act_dim_shooter, device)
        print("[INFO] No shooter API URL — using zero policy.")

    if cfg.goalkeeper_api:
        goalkeeper_policy = ApiPolicy(cfg.goalkeeper_api, act_dim_goalkeeper, device)
    else:
        goalkeeper_policy = _ZeroPolicy(act_dim_goalkeeper, device)
        print("[INFO] No goalkeeper API URL — using zero policy.")

    env = RslRlVecEnvWrapper(env_base, clip_actions=100.0)

    # -- Video recording (optional) -------------------------------------------
    if cfg.video and cfg.headless:
        from mjlab.utils.wrappers import VideoRecorder

        video_folder = Path("videos") / "compete"
        video_folder.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Recording video to: {video_folder}")
        env = VideoRecorder(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )

    # -- Run ------------------------------------------------------------------
    if cfg.headless:
        if cfg.num_trials <= 0:
            print("[WARN] --headless without --num-trials; nothing to evaluate.")
        else:
            run_headless_eval(
                cfg.num_trials, env, env_base, shooter_policy, goalkeeper_policy,
            )
    else:
        if cfg.num_trials > 0:
            print("[INFO] --num-trials set without --headless; launching viewer.")
        combined = CombinedPolicy(
            shooter_policy, goalkeeper_policy, env_base, device,
        )
        if cfg.viewer == "auto":
            has_display = bool(
                os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
            )
            viewer_type = "native" if has_display else "viser"
        else:
            viewer_type = cfg.viewer
        run_viewer(viewer_type, env, combined)

    env.close()


def main() -> None:
    import mjlab.tasks  # noqa: F401
    import src.tasks    # noqa: F401

    args = tyro.cli(CompeteConfig, prog="compete")
    run_compete(args)


if __name__ == "__main__":
    main()

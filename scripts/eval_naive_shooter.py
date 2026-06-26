"""Evaluate shooter — Stage II-compatible obs, zero-agent or trained checkpoint.

Scene: physical goal at origin, robot at penalty spot (-5,0,0.8),
ball at (-4,0,0.1).  Robot root pose is randomized each episode (±0.5m xy, ±0.5rad yaw).

Metrics (matching HumanoidSoccer Section IV-B):
  - Success Rate: ball crosses goal line inside the frame
  - Kick Accuracy: cosine similarity between ball velocity dir and ball→goal-center dir
  - Kick Speed: ball speed when it first exceeds 1 m/s

Usage:
  # Interactive viewer (zero agent)
  python scripts/eval_naive_shooter.py

  # Trained checkpoint
  python scripts/eval_naive_shooter.py --checkpoint logs/rsl_rl/g1_soccer/<run>/model_6000.pt

  # Headless multi-trial
  python scripts/eval_naive_shooter.py --headless --num-trials 50
  python scripts/eval_naive_shooter.py --headless --num-trials 50 --checkpoint <path>

  # Video
  python scripts/eval_naive_shooter.py --video --video-length 500
"""

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from src.tasks.soccer.config.g1.rl_cfg import (
  SoccerRecurrentRunner,
  unitree_g1_soccer_recurrent_runner_cfg,
)

# Goal geometry (motion-local coords: goal at (0, -5, 0), rotated 90° about Z
# so opening faces ±y.  Posts along x-axis, crossbar at z=1.8.)
_GOAL_HALF_WIDTH = 1.5   # m — half-width in x-direction
_GOAL_HEIGHT = 1.8       # m — crossbar height
_GOAL_Y = -5.0           # goal plane y-position
_GOAL_CENTER = (0.0, -5.0, 0.9)  # center of goal opening
_KICK_SPEED_THRESHOLD = 1.0  # m/s
_EVAL_SEEDS = (202606, 2810, 42)


@dataclass
class EvalConfig:
  video: bool = False
  video_length: int = 500
  video_height: int = 480
  video_width: int = 640
  viewer: str = "auto"
  device: str | None = None
  checkpoint: str | None = None
  seed: int = 2810
  headless: bool = False
  num_trials: int = 0

  task_id: str = "Eval-Shooter"


def _load_policy(checkpoint_path: str, env, device: str):
  print(f"[INFO] Loading policy from: {checkpoint_path}")
  agent_cfg = unitree_g1_soccer_recurrent_runner_cfg()
  runner = SoccerRecurrentRunner(env, asdict(agent_cfg), log_dir=None, device=device)
  runner.load(checkpoint_path)
  policy = runner.get_inference_policy(device=env.unwrapped.device)
  print("[INFO] Policy loaded successfully.")
  return policy


class ZeroPolicy:
  def __init__(self, action_shape, device):
    self._action = torch.zeros(action_shape, device=device)
  def __call__(self, obs):
    del obs
    return self._action
  def reset(self):
    pass


# -- Metrics ------------------------------------------------------------------

def _is_goal(ball_pos: torch.Tensor) -> bool:
  """Ball has crossed the goal plane (y=-5) inside the frame."""
  x, y, z = ball_pos[0].item(), ball_pos[1].item(), ball_pos[2].item()
  return y <= _GOAL_Y and abs(x) <= _GOAL_HALF_WIDTH and z <= _GOAL_HEIGHT


def _kick_accuracy(ball_vel: torch.Tensor, ball_pos: torch.Tensor) -> float:
  v_xy = ball_vel[:2]
  target_xy = torch.tensor([_GOAL_CENTER[0] - ball_pos[0].item(),
                             _GOAL_CENTER[1] - ball_pos[1].item()], dtype=torch.float32)
  v_norm = torch.norm(v_xy)
  t_norm = torch.norm(target_xy)
  if v_norm < 1e-6 or t_norm < 1e-6:
    return 0.0
  return float(torch.dot(v_xy / v_norm, target_xy / t_norm))


def run_trial(env, policy, max_steps: int = 500) -> dict:
  obs = env.reset()
  if isinstance(obs, tuple):
    obs = obs[0]
  if hasattr(policy, "reset"):
    policy.reset()

  ball = env.unwrapped.scene["ball"]
  kicked = False
  kick_speed = 0.0
  kick_accuracy_val = 0.0
  goal_scored = False
  ball_final_y = 0.0
  steps = 0

  for _ in range(max_steps):
    with torch.inference_mode():
      action = policy(obs)
    result = env.step(action)
    obs = result[0]
    terminated = bool(result[2].item()) if hasattr(result[2], "item") else bool(result[2])
    steps += 1

    ball_pos = ball.data.root_link_pos_w[0].cpu()
    ball_vel = ball.data.root_link_vel_w[0, :3].cpu()
    speed = float(torch.norm(ball_vel))

    if not kicked and speed > _KICK_SPEED_THRESHOLD:
      kicked = True
      kick_speed = speed
      kick_accuracy_val = _kick_accuracy(ball_vel, ball_pos)

    if _is_goal(ball_pos):
      goal_scored = True

    if terminated:
      break

    ball_final_y = float(ball_pos[1])

  return {
    "goal": goal_scored, "kick_speed": kick_speed,
    "kick_accuracy": kick_accuracy_val, "ball_final_y": ball_final_y,
    "steps": steps, "terminated": terminated,
  }


def run_headless_eval(cfg: EvalConfig, env, policy):
  if cfg.num_trials <= 0:
    print("[WARN] --headless without --num-trials: nothing to evaluate.")
    return None
  print(f"\n[INFO] Running {cfg.num_trials} headless eval trials (seed={cfg.seed})...\n")
  results = []
  goals = 0
  accuracies = []
  kick_speeds = []

  for trial in range(cfg.num_trials):
    stats = run_trial(env, policy)
    results.append(stats)
    if stats["goal"]:
      goals += 1
    if stats["kick_accuracy"] > 0:
      accuracies.append(stats["kick_accuracy"])
    if stats["kick_speed"] > 0:
      kick_speeds.append(stats["kick_speed"])

    interval = 1 if cfg.num_trials <= 10 else (cfg.num_trials // 10)
    if (trial + 1) % interval == 0 or trial == 0:
      print(f"  Trial {trial+1:3d}/{cfg.num_trials}: "
            f"goal={stats['goal']}, speed={stats['kick_speed']:.2f}, "
            f"acc={stats['kick_accuracy']:.3f}, steps={stats['steps']}, "
            f"term={stats['terminated']}")

  total = cfg.num_trials
  success_rate = goals / total * 100 if total > 0 else 0
  mean_acc = float(np.mean(accuracies)) if accuracies else 0.0
  std_acc = float(np.std(accuracies)) if accuracies else 0.0
  mean_speed = float(np.mean(kick_speeds)) if kick_speeds else 0.0
  ball_past = sum(1 for r in results if r["ball_final_y"] <= _GOAL_Y)

  print(f"\n{'='*55}")
  print(f"  Eval Summary ({total} trials)")
  print(f"{'='*55}")
  print(f"  Success Rate:        {goals}/{total} = {success_rate:.1f}%")
  print(f"  Kick Accuracy (cos): {mean_acc:.4f} ± {std_acc:.4f}")
  print(f"  Mean Kick Speed:     {mean_speed:.2f} m/s")
  print(f"  Ball past goal line: {ball_past}/{total}")
  print(f"{'='*55}\n")
  return {
    "seed": cfg.seed,
    "total": total,
    "goals": goals,
    "success_rate": success_rate,
    "mean_acc": mean_acc,
    "std_acc": std_acc,
    "mean_speed": mean_speed,
    "ball_past": ball_past,
  }


# -- Viewer -------------------------------------------------------------------

def run_viewer(cfg: EvalConfig, env, policy):
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    viewer_type = "native" if has_display else "viser"
  else:
    viewer_type = cfg.viewer

  if viewer_type == "native":
    NativeMujocoViewer(env, policy).run()
  elif viewer_type == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer: {viewer_type}")


# -- Main ---------------------------------------------------------------------

def run_eval(cfg: EvalConfig):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(cfg.task_id, play=False)
  env_cfg.seed = cfg.seed
  env_cfg.scene.num_envs = 1
  env_cfg.viewer.height = cfg.video_height
  env_cfg.viewer.width = cfg.video_width

  actor_terms = list(env_cfg.observations["actor"].terms.keys())
  print(f"Task: {cfg.task_id}")
  print(f"Actor obs ({len(actor_terms)} terms): {actor_terms}")
  print(f"Terminations: {list(env_cfg.terminations.keys())}")
  print(f"Episode length: {env_cfg.episode_length_s}s")

  render_mode = "rgb_array" if cfg.video else None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if cfg.video:
    video_folder = Path("videos") / "eval"
    video_folder.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Recording video to: {video_folder}")
    env = VideoRecorder(env, video_folder=video_folder,
                        step_trigger=lambda step: step == 0,
                        video_length=cfg.video_length, disable_logger=True)

  env = RslRlVecEnvWrapper(env, clip_actions=100.0)

  # Policy.
  if cfg.checkpoint:
    policy = _load_policy(cfg.checkpoint, env, device)
  else:
    action_shape = env.unwrapped.action_space.shape
    policy = ZeroPolicy(action_shape, device)
    print("[INFO] Using zero-agent fallback (no checkpoint provided).")

  obs_space = env.unwrapped.single_observation_space
  print(f"Actor obs dim:  {obs_space.spaces['actor'].shape}")
  print(f"Critic obs dim: {obs_space.spaces['critic'].shape}")

  if cfg.headless:
    result = run_headless_eval(cfg, env, policy)
  else:
    if cfg.num_trials > 0:
      print("[INFO] --num-trials set without --headless; running viewer.")
    run_viewer(cfg, env, policy)

  env.close()
  return result if cfg.headless else None


def run_multi_seed_eval(cfg: EvalConfig):
  results = []
  for seed in _EVAL_SEEDS:
    print(f"\n[INFO] ===== Evaluating seed {seed} =====")
    cfg.seed = seed
    cfg.video = False
    result = run_eval(cfg)
    if result is not None:
      results.append(result)

  if not results:
    return

  print(f"\n{'='*72}")
  print("  Multi-Seed Eval Summary")
  print(f"{'='*72}")
  for r in results:
    print(
      f"  Seed {r['seed']:>6}: "
      f"success={r['goals']}/{r['total']} ({r['success_rate']:.1f}%), "
      f"acc={r['mean_acc']:.4f}±{r['std_acc']:.4f}, "
      f"speed={r['mean_speed']:.2f} m/s, "
      f"past_line={r['ball_past']}/{r['total']}"
    )

  totals = sum(r["total"] for r in results)
  goals = sum(r["goals"] for r in results)
  weighted_speed = sum(r["mean_speed"] * r["total"] for r in results) / totals
  weighted_acc = sum(r["mean_acc"] * r["total"] for r in results) / totals
  print("-" * 72)
  print(
    f"  Overall: success={goals}/{totals} ({goals / totals * 100:.1f}%), "
    f"mean_acc={weighted_acc:.4f}, mean_speed={weighted_speed:.2f} m/s"
  )
  print(f"{'='*72}\n")


def main():
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  args = tyro.cli(EvalConfig, prog="eval_naive_shooter")
  if args.headless:
    run_multi_seed_eval(args)
  else:
    run_eval(args)


if __name__ == "__main__":
  main()

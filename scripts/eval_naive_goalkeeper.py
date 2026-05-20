"""Evaluate naive goalkeeper — matches Humanoid-Goalkeeper paper observation space.

Runs the goalkeeper environment with zero agent, records video, and
reports observation dimensions, ball velocity randomization, and
domain randomization events.

Usage:
  python scripts/eval_naive_goalkeeper.py
  python scripts/eval_naive_goalkeeper.py --video --video-length=500
  python scripts/eval_naive_goalkeeper.py --no-video
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass
class EvalConfig:
  video: bool = False
  video_length: int = 500  # steps (10s at 50Hz)
  video_height: int = 480
  video_width: int = 640
  viewer: str = "auto"
  device: str | None = None

  # Internal
  task_id: str = "Eval-Naive-Goalkeeper"


def run_eval(cfg: EvalConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(cfg.task_id, play=False)
  env_cfg.scene.num_envs = 1
  env_cfg.viewer.height = cfg.video_height
  env_cfg.viewer.width = cfg.video_width

  # Print observation info.
  actor_terms = list(env_cfg.observations["actor"].terms.keys())
  critic_terms = list(env_cfg.observations["critic"].terms.keys())
  events = list(env_cfg.events.keys())
  print(f"Task: {cfg.task_id}")
  print(f"Actor obs  ({len(actor_terms)} terms): {actor_terms}")
  print(f"Critic obs ({len(critic_terms)} terms): {critic_terms}")
  print(f"Events     ({len(events)}): {events}")
  print(f"Episode length: {env_cfg.episode_length_s}s")

  # Show ball velocity randomization params from settings.
  from src.tasks.soccer.config.soccer_settings import SETTINGS

  gv = SETTINGS.goalkeeper_ball_vel
  print(f"Ball speed:  [{gv.speed_min}, {gv.speed_max}] m/s")
  print(f"Ball pitch:  [{gv.pitch_min_deg}, {gv.pitch_max_deg}] deg")
  print(f"Goal margin: {gv.goal_margin} m")
  print(
    f"Yaw spread:  {gv.goal_margin:.1f} deg (auto-computed from goal geometry)"
  )

  render_mode = "rgb_array" if cfg.video else None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if cfg.video:
    video_folder = Path("videos") / "eval"
    video_folder.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Recording video to: {video_folder}")
    env = VideoRecorder(
      env,
      video_folder=video_folder,
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  # Wrap with RSL-RL wrapper and zero-agent policy.
  env = RslRlVecEnvWrapper(env, clip_actions=100.0)

  action_shape = env.unwrapped.action_space.shape

  class ZeroPolicy:
    def __call__(self, obs):
      del obs
      return torch.zeros(action_shape, device=env.unwrapped.device)

    def reset(self):
      pass

  policy = ZeroPolicy()

  # Select viewer.
  if cfg.viewer == "auto":
    has_display = bool(
      os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    viewer_type = "native" if has_display else "viser"
  else:
    viewer_type = cfg.viewer

  if viewer_type == "native":
    NativeMujocoViewer(env, policy).run()
  elif viewer_type == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer: {viewer_type}")

  env.close()


def main():
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  all_tasks = list_tasks()
  eval_tasks = [t for t in all_tasks if "Eval" in t]
  if not eval_tasks:
    print("No eval tasks registered. Run: import src.tasks")
    return

  args = tyro.cli(EvalConfig, prog="eval_naive_goalkeeper")
  run_eval(args)


if __name__ == "__main__":
  main()

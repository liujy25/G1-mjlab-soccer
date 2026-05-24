"""Evaluate goalkeeper — matches Humanoid-Goalkeeper paper eval protocol.

Runs the goalkeeper environment with a trained policy (or zero-agent fallback),
records video, and reports observation dimensions. In headless mode, runs
multiple trials and collects interception statistics.

Ball trajectory: 6-region parabolic model (matching paper's assign_ball_states).
Each episode randomly selects a region and samples a ball trajectory.

Usage:
  # Interactive viewer (zero agent)
  python scripts/eval_naive_goalkeeper.py

  # Interactive viewer (trained policy)
  python scripts/eval_naive_goalkeeper.py --checkpoint logs/rsl_rl/g1_soccer/model_5000.pt

  # Headless multi-trial eval with stats
  python scripts/eval_naive_goalkeeper.py --headless --num-trials=50
  python scripts/eval_naive_goalkeeper.py --headless --num-trials=500 --checkpoint <path>

  # With video
  python scripts/eval_naive_goalkeeper.py --video --video-length=300
"""

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg
from src.tasks.soccer.config.g1.rl_cfg import GoalkeeperRunner, unitree_g1_goalkeeper_ppo_runner_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

# ----- Goal geometry (goal at (-0.5, 0, 0) behind G1, 3.0m × 1.8m) -----
_GOAL_X = -0.5
_GOAL_HALF_WIDTH = 1.5    # m — half of goal width in y
_GOAL_HEIGHT = 1.8         # m — crossbar height


@dataclass
class EvalConfig:
  video: bool = False
  video_length: int = 300  # steps (6s at 50Hz, double the 3s episode)
  video_height: int = 480
  video_width: int = 640
  viewer: str = "auto"  # "auto", "native", "viser"
  device: str | None = None
  checkpoint: str | None = None  # path to .pt checkpoint file
  seed: int = 2810
  headless: bool = False   # run without viewer, collect stats
  num_trials: int = 0      # number of eval episodes (>0 implies headless)

  # Internal
  task_id: str = "Eval-Goalkeeper"


def _load_policy(checkpoint_path: str, env, device: str):
  """Load a Goalkeeper checkpoint using GoalkeeperRunner directly.

  Uses GoalkeeperRunner + unitree_g1_goalkeeper_ppo_runner_cfg regardless of
  the task's registered runner (which is None in the simplified template).
  Detects reference HIMPPO checkpoints (which store a single model_state_dict
  for the unified ActorCritic) and loads them directly, bypassing mjlab's
  legacy migration which would convert keys to MLPModel format.
  """
  print(f"[INFO] Loading policy from: {checkpoint_path}")
  loaded = torch.load(checkpoint_path, map_location=device)

  agent_cfg = unitree_g1_goalkeeper_ppo_runner_cfg()
  runner = GoalkeeperRunner(env, asdict(agent_cfg), device=device)

  if "model_state_dict" in loaded and hasattr(runner.alg.actor, "history_encoder"):
    print("[INFO] Detected HIMPPO ActorCritic checkpoint — loading directly.")
    actor_state = {k: v for k, v in loaded["model_state_dict"].items() if not k.startswith("critic.")}
    runner.alg.actor.load_state_dict(actor_state, strict=False)
    print("[INFO] Policy loaded successfully.")
  else:
    runner.load(checkpoint_path, load_cfg={"actor": True})
    print("[INFO] Policy loaded successfully.")

  policy = runner.get_inference_policy(device=env.unwrapped.device)
  return policy


def _make_zero_policy(env, device):
  """Return a zero-action policy for baseline evaluation."""
  act_dim = env.num_actions
  class ZeroPolicy:
    def __call__(self, obs):
      del obs
      return torch.zeros(1, act_dim, device=device)
    def reset(self):
      pass
  return ZeroPolicy()


# ----- Eval metric: ball must not enter the goal -----


def _ball_entered_goal(ball_pos: torch.Tensor) -> bool:
  """Ball has crossed the goal plane (x=-0.5) inside the goal frame."""
  x, y, z = ball_pos[0].item(), ball_pos[1].item(), ball_pos[2].item()
  return x <= _GOAL_X and abs(y) <= _GOAL_HALF_WIDTH and z <= _GOAL_HEIGHT


def run_trial(env, policy, max_steps: int = 150) -> dict:
  """Run one eval episode and return whether the ball entered the goal."""
  obs = env.reset()
  if isinstance(obs, tuple):
    obs = obs[0]

  ball = env.unwrapped.scene["ball"]
  ball_entered = False
  steps = 0

  for _ in range(max_steps):
    with torch.inference_mode():
      action = policy(obs)
    result = env.step(action)
    obs = result[0]
    terminated = bool(result[2].item())
    steps += 1

    ball_pos = ball.data.root_link_pos_w[0].cpu()
    if _ball_entered_goal(ball_pos):
      ball_entered = True

    if terminated:
      break

  return {"ball_entered_goal": ball_entered, "steps": steps, "terminated": terminated}


# ----- Headless multi-trial eval -----


def run_headless_eval(cfg: EvalConfig, env, policy):
  """Run multiple trials headless and print summary statistics."""
  if cfg.num_trials <= 0:
    print("[WARN] --headless without --num-trials: nothing to evaluate.")
    return
  print(f"\n[INFO] Running {cfg.num_trials} headless eval trials...\n")

  blocked_count = 0

  for trial in range(cfg.num_trials):
    stats = run_trial(env, policy)
    if not stats["ball_entered_goal"]:
      blocked_count += 1

    print_interval = 1 if cfg.num_trials <= 10 else (cfg.num_trials // 10)
    if (trial + 1) % print_interval == 0 or trial == 0:
      print(
        f"  Trial {trial + 1:3d}/{cfg.num_trials}: "
        f"blocked={not stats['ball_entered_goal']}, "
        f"steps={stats['steps']}"
      )

  total = cfg.num_trials
  success_rate = blocked_count / total * 100 if total > 0 else 0

  print(f"\n{'='*55}")
  print(f"  Eval Summary ({total} trials)")
  print(f"{'='*55}")
  print(f"  Block Rate:  {blocked_count}/{total} = {success_rate:.1f}%")
  print(f"{'='*55}\n")


# ----- Main -----


def run_eval(cfg: EvalConfig):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(cfg.task_id, play=False)
  env_cfg.scene.num_envs = 1
  env_cfg.viewer.height = cfg.video_height
  env_cfg.viewer.width = cfg.video_width

  actor_terms = list(env_cfg.observations["actor"].terms.keys())
  critic_terms = list(env_cfg.observations["critic"].terms.keys())
  events = list(env_cfg.events.keys())
  term_names = list(env_cfg.terminations.keys())
  actor_hist = env_cfg.observations["actor"].history_length

  print(f"Task: {cfg.task_id}")
  print(f"Actor obs  ({len(actor_terms)} terms × {actor_hist} history): {actor_terms}")
  print(f"Critic obs ({len(critic_terms)} terms): {critic_terms}")
  print(f"Terminations ({len(term_names)}): {term_names}")
  print(f"Events     ({len(events)}): {events}")
  print(f"Episode length: {env_cfg.episode_length_s}s")
  print(f"Obs noise: {env_cfg.observations['actor'].enable_corruption}")

  from src.tasks.soccer.config.soccer_settings import SETTINGS

  print(f"\nBall trajectory ({len(SETTINGS.goalkeeper_regions)} regions, parabolic model):")
  bt = SETTINGS.ball_trajectory
  print(f"  start: [{bt.ball_start_distance[0]}, {bt.ball_start_distance[1]}] m")
  print(f"  end:   [{bt.ball_end_distance[0]}, {bt.ball_end_distance[1]}] m")
  print(f"  t:     [{bt.t_flight[0]}, {bt.t_flight[1]}] s")

  region_names = [
    "Right-Mid", "Left-Mid", "Right-Up", "Left-Up", "Right-Low", "Left-Low",
  ]
  for i, r in enumerate(SETTINGS.goalkeeper_regions):
    print(f"  Region {i} ({region_names[i]}): h={r.height}, w={r.width}")

  render_mode = "rgb_array" if cfg.video else None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if cfg.video:
    video_folder = Path("videos") / "eval"
    video_folder.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFO] Recording video to: {video_folder}")
    env = VideoRecorder(
      env,
      video_folder=video_folder,
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=100.0)

  if cfg.checkpoint:
    policy = _load_policy(cfg.checkpoint, env, device)
  else:
    policy = _make_zero_policy(env, device)
    print("[INFO] Using zero-agent fallback (no checkpoint provided).")

  obs_space = env.unwrapped.single_observation_space
  print(f"\nRuntime shapes:")
  print(f"  Actor obs dim:  {obs_space.spaces['actor'].shape}")
  print(f"  Critic obs dim: {obs_space.spaces['critic'].shape}")
  print(f"  Action dim:     {env.num_actions}")

  if cfg.headless:
    run_headless_eval(cfg, env, policy)
  else:
    if cfg.num_trials > 0:
      print("[INFO] --num-trials is set but --headless is not; "
            "running viewer (use --headless for batch eval stats).")

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

"""Evaluate shooter — matches HumanoidSoccer paper observation space.

Runs the shooter environment with a trained policy (or zero-agent fallback),
records video, and reports observation dimensions.  In headless mode, runs
multiple trials and collects goal / kick-accuracy statistics (matching the
paper's eval protocol in Section IV-B).

Motion command: loads .npz kick motion files from src/assets/soccer/motions
and provides per-frame reference trajectories (O^ref_t) to the policy.

Ball position is fixed (per settings.yaml); robot root pose is randomized
each episode.

Usage:
  # Interactive viewer (zero agent)
  python scripts/eval_naive_shooter.py

  # Interactive viewer (trained policy)
  python scripts/eval_naive_shooter.py --checkpoint logs/rsl_rl/g1_soccer/model_5000.pt

  # Headless multi-trial eval with stats
  python scripts/eval_naive_shooter.py --headless --num-trials=50
  python scripts/eval_naive_shooter.py --headless --num-trials=50 --checkpoint <path>

  # With video
  python scripts/eval_naive_shooter.py --video --video-length=500
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import (RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg,
                      RslRlVecEnvWrapper)
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from rsl_rl.runners import OnPolicyRunner

from src.tasks.soccer.motion_data import MotionCommand, MotionDataset

_MOTION_DIR = Path(__file__).parent.parent / "src" / "assets" / "soccer" / "motions"

# Goal geometry (from settings.yaml goal.width=3.0, goal.height=1.8).
_GOAL_HALF_WIDTH = 1.5   # m
_GOAL_HEIGHT = 1.8        # m
_GOAL_X = 0.0             # goal-line x in world frame
_GOAL_CENTER = (0.0, 0.0, 0.9)

# Kick detection: ball speed must exceed this to count as "kicked".
_KICK_SPEED_THRESHOLD = 1.0  # m/s


@dataclass
class EvalConfig:
  video: bool = False
  video_length: int = 500  # steps (10s at 50Hz)
  video_height: int = 480
  video_width: int = 640
  viewer: str = "auto"  # "auto", "native", "viser"
  device: str | None = None
  checkpoint: str | None = None  # path to .pt checkpoint file
  motion_dir: str = str(_MOTION_DIR)
  seed: int = 2810
  headless: bool = False   # run without viewer, collect stats
  num_trials: int = 0      # number of eval episodes (>0 implies headless)

  # Internal
  task_id: str = "Eval-Naive-Shooter"


def _make_agent_cfg():
  """Minimal PPO config sufficient for loading a policy checkpoint."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_soccer_eval",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )


def _load_policy(checkpoint_path: str, env, device: str):
  """Load a PPO checkpoint and return the inference policy."""
  print(f"[INFO] Loading policy from: {checkpoint_path}")
  agent_cfg = _make_agent_cfg()
  runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
  runner.load(checkpoint_path)
  policy = runner.get_inference_policy(device=env.unwrapped.device)
  print("[INFO] Policy loaded successfully.")
  return policy


def _make_zero_policy(action_shape: torch.Size, device):
  """Return a zero-action policy for baseline evaluation."""
  class ZeroPolicy:
    def __call__(self, obs):
      del obs
      return torch.zeros(action_shape, device=device)
    def reset(self):
      pass
  return ZeroPolicy()


# ---------------------------------------------------------------------------
# Headless multi-trial eval with paper metrics
# ---------------------------------------------------------------------------

def _is_goal(ball_pos: torch.Tensor) -> bool:
  """Check whether the ball has crossed the goal line inside the frame."""
  x, y, z = ball_pos[0].item(), ball_pos[1].item(), ball_pos[2].item()
  return (x >= _GOAL_X and abs(y) <= _GOAL_HALF_WIDTH and z <= _GOAL_HEIGHT)


def _kick_accuracy(ball_vel: torch.Tensor, ball_pos: torch.Tensor) -> float:
  """Cosine similarity between ball velocity (xy) and ball→goal-center (xy)."""
  v_xy = ball_vel[:2]
  target_xy = torch.tensor(
    [_GOAL_CENTER[0] - ball_pos[0].item(),
     _GOAL_CENTER[1] - ball_pos[1].item()],
    dtype=torch.float32,
  )
  v_norm = torch.norm(v_xy)
  t_norm = torch.norm(target_xy)
  if v_norm < 1e-6 or t_norm < 1e-6:
    return 0.0
  return float(torch.dot(v_xy / v_norm, target_xy / t_norm))


def run_trial(env, policy, motion_cmd, max_steps: int = 500) -> dict:
  """Run one eval episode and return stats dict.

  Keys: goal (bool), kick_speed (float), kick_accuracy (float),
        ball_final_x (float), steps (int), terminated (bool)
  """
  motion_cmd.reset()
  obs = env.reset()
  if isinstance(obs, tuple):
    obs = obs[0]

  ball = env.unwrapped.scene["ball"]
  kicked = False
  kick_speed = 0.0
  kick_accuracy_val = 0.0
  goal_scored = False
  ball_final_x = -4.0  # nominal start position
  steps = 0

  for _ in range(max_steps):
    motion_cmd.step()
    with torch.inference_mode():
      action = policy(obs)
    result = env.step(action)
    obs = result[0]
    # RSL-RL wrapper returns (obs, rewards, dones, infos) — 4-tuple.
    terminated = bool(result[2].item()) if hasattr(result[2], 'item') else bool(result[2])
    steps += 1

    # Read ball state.
    ball_pos = ball.data.root_link_pos_w[0].cpu()
    ball_vel = ball.data.root_link_vel_w[0, :3].cpu()
    speed = float(torch.norm(ball_vel))

    if not kicked and speed > _KICK_SPEED_THRESHOLD:
      kicked = True
      kick_speed = speed
      kick_accuracy_val = _kick_accuracy(ball_vel, ball_pos)

    if _is_goal(ball_pos):
      goal_scored = True

    ball_final_x = float(ball_pos[0])

    if terminated:
      break

  return {
    "goal": goal_scored,
    "kick_speed": kick_speed,
    "kick_accuracy": kick_accuracy_val,
    "ball_final_x": ball_final_x,
    "steps": steps,
    "terminated": terminated,
  }


def run_headless_eval(cfg: EvalConfig, env, policy, motion_cmd):
  """Run multiple trials headless and print summary statistics."""
  if cfg.num_trials <= 0:
    print("[WARN] --headless without --num-trials: nothing to evaluate.")
    return
  print(f"\n[INFO] Running {cfg.num_trials} headless eval trials...\n")
  results = []
  goals = 0
  accuracies = []
  kick_speeds = []

  for trial in range(cfg.num_trials):
    stats = run_trial(env, policy, motion_cmd)
    results.append(stats)
    if stats["goal"]:
      goals += 1
    if stats["kick_accuracy"] > 0:
      accuracies.append(stats["kick_accuracy"])
    if stats["kick_speed"] > 0:
      kick_speeds.append(stats["kick_speed"])

    print_interval = 1 if cfg.num_trials <= 10 else (cfg.num_trials // 10)
    if (trial + 1) % print_interval == 0 or trial == 0:
      print(f"  Trial {trial+1:3d}/{cfg.num_trials}: "
            f"goal={stats['goal']}, "
            f"speed={stats['kick_speed']:.2f}, "
            f"acc={stats['kick_accuracy']:.3f}, "
            f"steps={stats['steps']}, "
            f"term={stats['terminated']}")

  # Summary.
  success_rate = goals / cfg.num_trials * 100 if cfg.num_trials > 0 else 0
  mean_acc = float(np.mean(accuracies)) if accuracies else 0.0
  std_acc = float(np.std(accuracies)) if accuracies else 0.0
  mean_speed = float(np.mean(kick_speeds)) if kick_speeds else 0.0
  ball_past_goal = sum(1 for r in results if r["ball_final_x"] >= _GOAL_X)

  print(f"\n{'='*55}")
  print(f"  Eval Summary ({cfg.num_trials} trials)")
  print(f"{'='*55}")
  print(f"  Success Rate:        {goals}/{cfg.num_trials} = {success_rate:.1f}%")
  print(f"  Kick Accuracy (cos): {mean_acc:.4f} ± {std_acc:.4f}")
  print(f"  Mean Kick Speed:     {mean_speed:.2f} m/s")
  print(f"  Ball past goal line: {ball_past_goal}/{cfg.num_trials}")
  print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Interactive viewer mode
# ---------------------------------------------------------------------------

def run_viewer(cfg: EvalConfig, env, policy, motion_cmd):
  """Run interactive viewer with motion-command stepping."""
  # Select viewer.
  if cfg.viewer == "auto":
    has_display = bool(
      os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    viewer_type = "native" if has_display else "viser"
  else:
    viewer_type = cfg.viewer

  # Wrap step: advance motion command before each physics step,
  # reset on episode boundaries.
  _original_step = env.step

  def _step_with_motion(action):
    motion_cmd.step()
    result = _original_step(action)
    try:
      ep_len = env.unwrapped.episode_length_buf.item()
    except Exception:
      ep_len = -1
    if ep_len == 0:
      motion_cmd.reset()
    return result

  env.step = _step_with_motion

  if viewer_type == "native":
    NativeMujocoViewer(env, policy).run()
  elif viewer_type == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer: {viewer_type}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval(cfg: EvalConfig):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(cfg.task_id, play=False)
  env_cfg.scene.num_envs = 1
  env_cfg.viewer.height = cfg.video_height
  env_cfg.viewer.width = cfg.video_width

  # Print env info.
  actor_terms = list(env_cfg.observations["actor"].terms.keys())
  events = list(env_cfg.events.keys())
  term_names = list(env_cfg.terminations.keys())
  print(f"Task: {cfg.task_id}")
  print(f"Actor obs  ({len(actor_terms)} terms): {actor_terms}")
  print(f"Terminations ({len(term_names)}): {term_names}")
  print(f"Events     ({len(events)}): {events}")
  print(f"Episode length: {env_cfg.episode_length_s}s")

  # Load motion dataset.
  print(f"[INFO] Loading motions from: {cfg.motion_dir}")
  motion_dataset = MotionDataset(cfg.motion_dir, device=device)
  print(f"[INFO] Loaded {motion_dataset.num_motions} motions, "
        f"{motion_dataset.joint_dim}-DoF, max {motion_dataset.max_frames} fr")
  for info in motion_dataset.infos:
    print(f"  {info.name}: {info.num_frames}f, kick={info.kick_leg}")

  render_mode = "rgb_array" if cfg.video else None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  # Attach motion command.
  motion_cmd = MotionCommand(motion_dataset, num_envs=1, device=device)
  env.motion_command = motion_cmd

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

  # Wrap with RSL-RL.
  env = RslRlVecEnvWrapper(env, clip_actions=100.0)

  # Policy.
  if cfg.checkpoint:
    policy = _load_policy(cfg.checkpoint, env, device)
  else:
    action_shape = env.unwrapped.action_space.shape
    policy = _make_zero_policy(action_shape, device)
    print("[INFO] Using zero-agent fallback (no checkpoint provided).")

  # --headless triggers multi-trial eval; otherwise always run viewer.
  if cfg.headless:
    run_headless_eval(cfg, env, policy, motion_cmd)
  else:
    if cfg.num_trials > 0:
      print("[INFO] --num-trials is set but --headless is not; "
            "running viewer (use --headless for batch eval stats).")
    run_viewer(cfg, env, policy, motion_cmd)

  env.close()


def main():
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  all_tasks = list_tasks()
  eval_tasks = [t for t in all_tasks if "Eval" in t]
  if not eval_tasks:
    print("No eval tasks registered. Run: import src.tasks")
    return

  args = tyro.cli(EvalConfig, prog="eval_naive_shooter")
  run_eval(args)


if __name__ == "__main__":
  main()

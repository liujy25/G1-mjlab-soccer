"""Reference policy server for Phase 2 tournament.

Implements the standard REST API that ``compete.py`` calls during cross-evaluation.

Receives raw MuJoCo state (both robots + ball) and computes its own observation
tensor.  Teams should customize ``compute_obs()`` to match their training setup.

  POST /act    - receive raw state, return action
  POST /reset  - reset policy hidden state and history buffer

Usage:
  # Shooter server
  python scripts/api_server.py --checkpoint shooter.pt --port 8000 --task shooter

  # Goalkeeper server
  python scripts/api_server.py --checkpoint goalkeeper.pt --port 8001 --task goalkeeper

Test with curl:
  curl -X POST http://localhost:8000/reset
  curl -X POST http://localhost:8000/act \\
       -H "Content-Type: application/json" \\
       -d '{"shooter":{"root_pos":[4,0,0.8],...},"goalkeeper":{...},"ball":{...}}'

-------------------------------------------------------------------------------
CUSTOMIZATION GUIDE
-------------------------------------------------------------------------------

Teams MUST customize ``compute_obs()`` to match their policy's observation
space.  The default implementation computes a standard proprioception + ball
observation.  If your policy uses different terms, scaling factors, reference
frames, or history length, update the function accordingly.
"""

from __future__ import annotations

from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Any

import torch
import tyro
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

# ---------------------------------------------------------------------------
# Default joint positions  (match training configs)
# ---------------------------------------------------------------------------

_SHOOTER_DEFAULT_JOINT_POS = torch.tensor([
    0.0, 0.0, 0.0,          # left/right hip, waist
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # left leg
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # right leg
    0.0, 0.0, 0.0,          # torso
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # left arm
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # right arm
])

_GK_DEFAULT_JOINT_POS = torch.tensor([
    0.0, 0.0, 0.0,          # left/right hip, waist
    -0.35, 0.7, -0.35, -0.25, 0.3, -0.1,    # left leg
    -0.35, 0.7, -0.35, -0.25, 0.3, -0.1,    # right leg
    0.0, 0.3, 0.0,          # torso
    0.8, 0.0, -1.6, 0.0, 0.5, 0.0, 0.0,   # left arm
    0.8, 0.0, -1.6, 0.0, 0.5, 0.0, 0.0,   # right arm
])

# ---------------------------------------------------------------------------
# Observation computation  (CUSTOMIZE: match your training observation space)
# ---------------------------------------------------------------------------

def compute_shooter_obs(raw_state: dict) -> torch.Tensor:
    """Compute shooter observation tensor from raw state.

    Default: proprioception + ball position (~100-D, no history).
    Replace with your own obs terms, scaling, and concatenation order.
    """
    s = raw_state["shooter"]
    ball = raw_state["ball"]

    root_quat = torch.tensor(s["root_quat"])
    root_ang_vel = torch.tensor(s["root_ang_vel"])
    joint_pos = torch.tensor(s["joint_pos"])
    joint_vel = torch.tensor(s["joint_vel"])
    ball_pos = torch.tensor(ball["pos"])
    root_pos = torch.tensor(s["root_pos"])
    last_action = torch.tensor(s["last_action"])

    # Projected gravity
    gravity_w = torch.tensor([0.0, 0.0, -1.0])
    projected_gravity = quat_apply(quat_inv(root_quat), gravity_w)

    # Base angular velocity in robot frame
    base_ang_vel = quat_apply(quat_inv(root_quat), root_ang_vel)

    # Joint positions relative to default
    joint_pos_rel = joint_pos - _SHOOTER_DEFAULT_JOINT_POS

    # Ball position in robot pelvis frame
    ball_pos_local = quat_apply(quat_inv(root_quat), ball_pos - root_pos)

    obs = torch.cat([
        base_ang_vel,           # 3
        projected_gravity,      # 3
        joint_pos_rel,          # 29
        joint_vel,              # 29
        last_action,            # 29
        ball_pos_local,         # 3
    ])
    return obs.unsqueeze(0)  # (1, obs_dim)


def compute_goalkeeper_obs(raw_state: dict) -> torch.Tensor:
    """Compute goalkeeper observation tensor from raw state (single frame).

    Default: matches ``eval_goalkeeper_cfg`` per-frame terms (96-D).
    Replace with your own obs terms, scaling, and concatenation order.
    """
    s = raw_state["goalkeeper"]
    ball = raw_state["ball"]

    root_quat = torch.tensor(s["root_quat"])
    root_ang_vel = torch.tensor(s["root_ang_vel"])
    joint_pos = torch.tensor(s["joint_pos"])
    joint_vel = torch.tensor(s["joint_vel"])
    ball_pos = torch.tensor(ball["pos"])
    root_pos = torch.tensor(s["root_pos"])
    last_action = torch.tensor(s["last_action"])

    # Projected gravity
    gravity_w = torch.tensor([0.0, 0.0, -1.0])
    projected_gravity = quat_apply(quat_inv(root_quat), gravity_w)

    # Angular velocity with GK scaling (×0.25, matching GK PD gain ratio)
    base_ang_vel = quat_apply(quat_inv(root_quat), root_ang_vel) * 0.25

    # Joint positions relative to GK default, GK-specific scaling
    joint_pos_rel = (joint_pos - _GK_DEFAULT_JOINT_POS) * 1.0

    # Joint velocities with GK scaling (×0.05)
    joint_vel_scaled = joint_vel * 0.05

    # Ball position in robot pelvis frame
    ball_pos_local = quat_apply(quat_inv(root_quat), ball_pos - root_pos)

    obs = torch.cat([
        ball_pos_local,         # 3
        base_ang_vel,           # 3
        projected_gravity,      # 3
        joint_pos_rel,          # 29
        joint_vel_scaled,       # 29
        last_action,            # 29
    ])
    return obs.unsqueeze(0)  # (1, 96)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ActResponse(BaseModel):
    action: list[list[float]]  # shape: [1, act_dim]


# ---------------------------------------------------------------------------
# Policy loading  (for model architecture; obs are computed server-side)
# ---------------------------------------------------------------------------

def _load_policy(checkpoint_path: str, task_id: str, device: str) -> Any:
    """Build env from task config, load checkpoint, return inference policy."""
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()

    env_cfg = load_env_cfg(task_id, play=False)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    actor_terms = list(env_cfg.observations["actor"].terms.keys())
    history_len = env_cfg.observations["actor"].history_length
    print(f"[INFO] Task: {task_id}")
    print(f"[INFO] Actor obs  ({len(actor_terms)} terms × {history_len} history): {actor_terms}")
    print(f"[INFO] Action dim: {env.num_actions}")

    if task_id == "Eval-Goalkeeper":
        from src.tasks.soccer.config.g1.rl_cfg import (
            GoalkeeperRunner,
            unitree_g1_goalkeeper_ppo_runner_cfg,
        )
        loaded = torch.load(checkpoint_path, map_location=device)
        agent_cfg = unitree_g1_goalkeeper_ppo_runner_cfg()
        runner = GoalkeeperRunner(env, asdict(agent_cfg), device=device)

        if "model_state_dict" in loaded and hasattr(runner.alg.actor, "history_encoder"):
            print("[INFO] Detected HIMPPO ActorCritic checkpoint — loading directly.")
            actor_state = {
                k: v
                for k, v in loaded["model_state_dict"].items()
                if not k.startswith("critic.")
            }
            runner.alg.actor.load_state_dict(actor_state, strict=False)
        else:
            runner.load(checkpoint_path, load_cfg={"actor": True})
    else:
        from src.tasks.soccer.config.g1.rl_cfg import (
            SoccerRecurrentRunner,
            unitree_g1_soccer_recurrent_runner_cfg,
        )
        agent_cfg = unitree_g1_soccer_recurrent_runner_cfg()
        runner = SoccerRecurrentRunner(
            env, asdict(agent_cfg), log_dir=None, device=device,
        )
        runner.load(checkpoint_path)

    policy = runner.get_inference_policy(device=device)
    print(f"[INFO] Policy loaded from: {checkpoint_path}")
    return policy, env


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(checkpoint_path: str, task_id: str, device: str) -> FastAPI:
    """Build the FastAPI app with a loaded policy and obs computer."""

    policy, env = _load_policy(checkpoint_path, task_id, device)
    is_gk = task_id == "Eval-Goalkeeper"
    history_len = 10 if is_gk else 1

    # History buffer for goalkeeper's multi-frame observation stack.
    history: deque[torch.Tensor] = deque(maxlen=history_len)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print(f"[INFO] Server ready — {task_id} policy on {device}")
        yield
        env.close()
        print("[INFO] Server shutting down.")

    app = FastAPI(title=f"CS2810 Phase 2 — {task_id}", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/act", response_model=ActResponse)
    async def act(req: dict):
        # Compute per-frame observation from raw state.
        raw_state = req
        if is_gk:
            frame = compute_goalkeeper_obs(raw_state)
        else:
            frame = compute_shooter_obs(raw_state)

        # Initialize history buffer on first frame after reset.
        if len(history) == 0:
            for _ in range(history_len):
                history.append(frame.clone())

        history.append(frame)

        # Build stacked observation: (1, history_len × frame_dim).
        stacked = torch.cat(list(history), dim=-1)

        with torch.inference_mode():
            action = policy({"actor": stacked})

        return ActResponse(action=action.cpu().tolist())

    @app.post("/reset")
    async def reset():
        policy.reset()
        history.clear()
        return {"status": "ok"}

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    checkpoint: str
    """Path to the policy checkpoint (.pt)."""
    port: int = 8000
    """Port to listen on."""
    task: str = "shooter"
    """Task type: 'shooter' or 'goalkeeper'."""
    host: str = "0.0.0.0"
    """Host to bind to."""
    device: str | None = None
    """Torch device (auto-detected if omitted)."""


def main():
    import src.tasks  # noqa: F401  — register eval tasks

    args = tyro.cli(ServerConfig, prog="api_server")

    task_id = "Eval-Shooter" if args.task == "shooter" else "Eval-Goalkeeper"
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    app = create_app(args.checkpoint, task_id, device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

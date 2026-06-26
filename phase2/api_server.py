"""Reference policy server for CS2810 Phase 2.

Students run one server per policy.  The tournament runner sends raw MuJoCo
state to ``POST /act`` and expects a 29-D action in response.

Usage:
  python phase2/api_server.py --checkpoint shooter.pt --port 8000 --task shooter
  python phase2/api_server.py --checkpoint goalkeeper.pt --port 8001 --task goalkeeper
"""

from __future__ import annotations

from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import tyro
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv


_SHOOTER_DEFAULT_JOINT_POS: torch.Tensor | None = None
_GK_DEFAULT_JOINT_POS: torch.Tensor | None = None
_PHASE2_CONFIG_PATH = Path(__file__).resolve().with_name("phase2_config.yaml")
_SHOOTER_TARGET_HEIGHT = 0.11
_SHOOTER_TARGET_LATERAL_RANGE = (0.5, 1.1)


def _set_default_joint_pos_from_env(env: ManagerBasedRlEnv, task_id: str) -> None:
    """Use the loaded task's robot defaults so API obs match the 29-DoF env."""
    global _SHOOTER_DEFAULT_JOINT_POS, _GK_DEFAULT_JOINT_POS
    default = env.scene["robot"].data.default_joint_pos[0].detach().cpu().to(torch.float32)
    if task_id == "Eval-Goalkeeper":
        _GK_DEFAULT_JOINT_POS = default
    else:
        _SHOOTER_DEFAULT_JOINT_POS = default


def _default_joint_pos_for(role: str, joint_pos: torch.Tensor) -> torch.Tensor:
    default = _GK_DEFAULT_JOINT_POS if role == "goalkeeper" else _SHOOTER_DEFAULT_JOINT_POS
    if default is None:
        raise RuntimeError(f"{role} default joint positions were not initialized from the env")
    default = default.to(device=joint_pos.device, dtype=joint_pos.dtype)
    if default.shape != joint_pos.shape:
        raise RuntimeError(
            f"{role} joint dimension mismatch: raw_state has {joint_pos.numel()} joints, "
            f"loaded task default has {default.numel()} joints"
        )
    return default


def _load_phase2_target_destination() -> torch.Tensor:
    """Return the fixed Phase 2 shooting target in world coordinates."""
    if not _PHASE2_CONFIG_PATH.exists():
        return torch.tensor([-0.5, 0.0, _SHOOTER_TARGET_HEIGHT], dtype=torch.float32)

    with _PHASE2_CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    goal_cfg = config.get("goal", {})
    goal_pos = goal_cfg.get("pos", [-0.5, 0.0, 0.0])
    x = float(goal_cfg.get("plane_x", goal_pos[0]))
    y = float(goal_pos[1])
    return torch.tensor([x, y, _SHOOTER_TARGET_HEIGHT], dtype=torch.float32)


class ShooterEvalObservationBuilder:
    """Build the same actor observation layout used by Eval-Shooter Stage II."""

    def __init__(self, env: ManagerBasedRlEnv):
        self.env = env
        self.device = env.device
        self.motion = env.command_manager.get_term("motion")
        self.base_target_destination_w = _load_phase2_target_destination().to(self.device)
        self.target_destination_w = self.base_target_destination_w.clone()
        self.target_lateral_sign: float | None = None
        self.target_lateral_abs: float | None = None
        self.target_motion_selected = False
        robot = env.scene["robot"]
        self.default_joint_vel = robot.data.default_joint_vel[0].detach().to(self.device)

    def reset(self) -> None:
        self.env.reset()
        self.target_destination_w = self.base_target_destination_w.clone()
        self.target_lateral_sign = None
        self.target_lateral_abs = None
        self.target_motion_selected = False

    def _tensor(self, values: Any) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def _select_random_motion(self) -> None:
        if self.target_motion_selected:
            return

        num_motions = len(self.motion.motion.motion_names)
        if num_motions <= 0:
            print("[WARN] No shooter motion found; keeping sampled motion.")
            self.target_motion_selected = True
            return

        picked = torch.randint(0, num_motions, (), device=self.device)
        self.motion.motion_idx[0] = picked
        self.motion.motion_length[0] = self.motion.motion.file_lengths[picked]
        self.motion.time_steps[0] = 0

        motion_name = self.motion.motion.motion_names[int(picked.detach().cpu())]
        kick_leg = int(self.motion.motion_kick_leg[picked].detach().cpu())
        kick_leg_name = "right" if kick_leg == 1 else "left"
        print(
            f"[INFO] Shooter random motion selected for target_y="
            f"{self.target_destination_w[1].item():.3f}: {motion_name} ({kick_leg_name} foot)"
        )
        self.target_motion_selected = True

    def _sample_target_destination(self) -> None:
        if self.target_lateral_sign is None:
            self.target_lateral_sign = (
                1.0 if torch.rand((), device=self.device).item() < 0.5 else -1.0
            )
            low, high = _SHOOTER_TARGET_LATERAL_RANGE
            self.target_lateral_abs = float(
                (low + (high - low) * torch.rand((), device=self.device)).item()
            )
            target_y = self.target_lateral_sign * self.target_lateral_abs
            side = "right" if self.target_lateral_sign > 0.0 else "left"
            print(
                f"[INFO] Shooter target_y sampled to {side} side: {target_y:.3f} "
                f"(abs={self.target_lateral_abs:.3f})"
            )

        self.target_destination_w = self.base_target_destination_w.clone()
        self.target_destination_w[1] = self.target_lateral_sign * self.target_lateral_abs
        self._select_random_motion()

    def compute(self, raw_state: dict) -> torch.Tensor:
        s = raw_state["shooter"]
        ball = raw_state["ball"]

        root_quat = self._tensor(s["root_quat"])
        root_ang_vel = self._tensor(s["root_ang_vel"])
        joint_pos = self._tensor(s["joint_pos"])
        joint_vel = self._tensor(s["joint_vel"])
        ball_pos = self._tensor(ball["pos"])
        root_pos = self._tensor(s["root_pos"])
        last_action = self._tensor(s["last_action"])

        gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)
        projected_gravity = quat_apply(quat_inv(root_quat), gravity_w)
        base_ang_vel = quat_apply(quat_inv(root_quat), root_ang_vel)
        joint_pos_rel = joint_pos - _default_joint_pos_for("shooter", joint_pos)
        joint_vel_rel = joint_vel - self.default_joint_vel.to(joint_vel.device)
        target_point_pos = quat_apply(quat_inv(root_quat), ball_pos - root_pos)
        self._sample_target_destination()
        target_destination_pos = quat_apply(
            quat_inv(root_quat), self.target_destination_w - root_pos
        )

        obs = torch.cat([
            self.motion.command[0],
            projected_gravity,
            self.motion.anchor_ang_vel_w[0],
            base_ang_vel,
            joint_pos_rel,
            joint_vel_rel,
            last_action,
            target_point_pos,
            target_destination_pos,
        ])
        return obs.unsqueeze(0)

    def advance(self) -> None:
        self.env.command_manager.compute(dt=self.env.step_dt)


def compute_goalkeeper_obs(raw_state: dict) -> torch.Tensor:
    """Compute a default goalkeeper observation from raw state.

    Teams should customize this function to match their own training setup.
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

    gravity_w = torch.tensor([0.0, 0.0, -1.0])
    projected_gravity = quat_apply(quat_inv(root_quat), gravity_w)
    base_ang_vel = quat_apply(quat_inv(root_quat), root_ang_vel) * 0.25
    joint_pos_rel = joint_pos - _default_joint_pos_for("goalkeeper", joint_pos)
    joint_vel_scaled = joint_vel * 0.05
    ball_pos_local = quat_apply(quat_inv(root_quat), ball_pos - root_pos)

    obs = torch.cat([
        ball_pos_local,
        base_ang_vel,
        projected_gravity,
        joint_pos_rel,
        joint_vel_scaled,
        last_action,
    ])
    return obs.unsqueeze(0)


class ActResponse(BaseModel):
    action: list[list[float]]


def _load_policy(checkpoint_path: str, task_id: str, device: str) -> Any:
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    env_cfg = load_env_cfg(task_id, play=False)
    env_cfg.scene.num_envs = 1
    env_base = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    _set_default_joint_pos_from_env(env_base, task_id)
    env = RslRlVecEnvWrapper(env_base, clip_actions=100.0)

    actor_terms = list(env_cfg.observations["actor"].terms.keys())
    history_len = env_cfg.observations["actor"].history_length
    action_dim = env_base.action_manager.get_term("joint_pos").action_dim
    print(f"[INFO] Task: {task_id}")
    print(f"[INFO] Actor obs ({len(actor_terms)} terms x {history_len} history): {actor_terms}")
    print(f"[INFO] Action dim: {action_dim}")

    if task_id == "Eval-Goalkeeper":
        from src.tasks.soccer.config.g1.rl_cfg import (
            GoalkeeperRunner,
            unitree_g1_goalkeeper_ppo_runner_cfg,
        )

        loaded = torch.load(checkpoint_path, map_location=device)
        agent_cfg = unitree_g1_goalkeeper_ppo_runner_cfg()
        runner = GoalkeeperRunner(env, asdict(agent_cfg), device=device)
        if "model_state_dict" in loaded and hasattr(runner.alg.actor, "history_encoder"):
            print("[INFO] Detected HIMPPO ActorCritic checkpoint.")
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
        runner = SoccerRecurrentRunner(env, asdict(agent_cfg), log_dir=None, device=device)
        runner.load(checkpoint_path)

    policy = runner.get_inference_policy(device=device)
    print(f"[INFO] Policy loaded from: {checkpoint_path}")
    return policy, env_base


def create_app(checkpoint_path: str, task_id: str, device: str) -> FastAPI:
    policy, env = _load_policy(checkpoint_path, task_id, device)
    is_gk = task_id == "Eval-Goalkeeper"
    history_len = 10 if is_gk else 1
    history: deque[torch.Tensor] = deque(maxlen=history_len)
    shooter_obs = None if is_gk else ShooterEvalObservationBuilder(env)
    if shooter_obs is not None:
        shooter_obs.reset()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print(f"[INFO] Server ready: {task_id} policy on {device}")
        yield
        env.close()
        print("[INFO] Server shutting down.")

    app = FastAPI(title=f"CS2810 Phase 2 - {task_id}", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/act", response_model=ActResponse)
    async def act(req: dict):
        frame = compute_goalkeeper_obs(req) if is_gk else shooter_obs.compute(req)
        frame = frame.to(device)
        if len(history) == 0:
            for _ in range(history_len):
                history.append(frame.clone())
        history.append(frame)
        stacked = torch.cat(list(history), dim=-1)

        with torch.inference_mode():
            action = policy({"actor": stacked})
        if shooter_obs is not None:
            shooter_obs.advance()
        return ActResponse(action=action.cpu().tolist())

    @app.post("/reset")
    async def reset():
        policy.reset()
        history.clear()
        if shooter_obs is not None:
            shooter_obs.reset()
        return {"status": "ok"}

    return app


@dataclass
class ServerConfig:
    checkpoint: str
    port: int = 8000
    task: str = "shooter"
    host: str = "0.0.0.0"
    device: str | None = None


def main() -> None:
    import src.tasks  # noqa: F401

    args = tyro.cli(ServerConfig, prog="phase2-api-server")
    if args.task not in {"shooter", "goalkeeper"}:
        raise ValueError("--task must be either 'shooter' or 'goalkeeper'")
    task_id = "Eval-Shooter" if args.task == "shooter" else "Eval-Goalkeeper"
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    app = create_app(args.checkpoint, task_id, device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

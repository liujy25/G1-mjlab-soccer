"""Unitree G1 soccer task registration."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  unitree_g1_shooter_env_cfg,
  unitree_g1_goalkeeper_env_cfg,
)
from .rl_cfg import unitree_g1_soccer_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-Naive-Shooter",
  env_cfg=unitree_g1_shooter_env_cfg(),
  play_env_cfg=unitree_g1_shooter_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_ppo_runner_cfg(),
  runner_cls=None,  # Uses default MjlabOnPolicyRunner
)

register_mjlab_task(
  task_id="Unitree-G1-Naive-Goalkeeper",
  env_cfg=unitree_g1_goalkeeper_env_cfg(),
  play_env_cfg=unitree_g1_goalkeeper_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_ppo_runner_cfg(),
  runner_cls=None,  # Uses default MjlabOnPolicyRunner
)

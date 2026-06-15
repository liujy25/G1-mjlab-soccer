"""Unitree G1 soccer task registration."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  unitree_g1_shooter_env_cfg,
  unitree_g1_goalkeeper_env_cfg,
)

from .training_env_cfgs import (
  unitree_g1_stage1_env_cfg,
  unitree_g1_stage2_env_cfg,
)
from .rl_cfg import (
  unitree_g1_soccer_ppo_runner_cfg,
  unitree_g1_soccer_recurrent_runner_cfg,
  SoccerRecurrentRunner,
)

# -- Soccer tasks -------------------------------------------------------------

register_mjlab_task(
  task_id="Unitree-G1-Shooter",
  env_cfg=unitree_g1_shooter_env_cfg(),
  play_env_cfg=unitree_g1_shooter_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_ppo_runner_cfg(),
  runner_cls=None,
)

register_mjlab_task(
  task_id="Unitree-G1-Goalkeeper",
  env_cfg=unitree_g1_goalkeeper_env_cfg(),
  play_env_cfg=unitree_g1_goalkeeper_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_ppo_runner_cfg(),
  runner_cls=None,
)

register_mjlab_task(
  task_id="Unitree-G1-Shooter-Stage1",
  env_cfg=unitree_g1_stage1_env_cfg(),
  play_env_cfg=unitree_g1_stage1_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_recurrent_runner_cfg(),
  runner_cls=SoccerRecurrentRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Shooter-Stage2",
  env_cfg=unitree_g1_stage2_env_cfg(),
  play_env_cfg=unitree_g1_stage2_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_recurrent_runner_cfg(),
  runner_cls=SoccerRecurrentRunner,
)
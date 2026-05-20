"""Register evaluation tasks for naive shooter and goalkeeper.

These tasks use observation spaces and domain randomization matching
the HumanoidSoccer and Humanoid-Goalkeeper papers respectively.
"""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.soccer.config.eval.eval_goalkeeper_cfg import eval_goalkeeper_env_cfg
from src.tasks.soccer.config.eval.eval_shooter_cfg import eval_shooter_env_cfg
from src.tasks.soccer.config.g1.rl_cfg import unitree_g1_soccer_ppo_runner_cfg

register_mjlab_task(
  task_id="Eval-Naive-Shooter",
  env_cfg=eval_shooter_env_cfg(play=False),
  play_env_cfg=eval_shooter_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_ppo_runner_cfg(),
  runner_cls=None,
)

register_mjlab_task(
  task_id="Eval-Naive-Goalkeeper",
  env_cfg=eval_goalkeeper_env_cfg(play=False),
  play_env_cfg=eval_goalkeeper_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_ppo_runner_cfg(),
  runner_cls=None,
)

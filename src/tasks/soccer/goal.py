"""Goal entity configuration for soccer tasks."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.entity import EntityCfg

GOAL_XML: Path = SRC_PATH / "assets" / "soccer" / "goal.xml"
assert GOAL_XML.exists()


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GOAL_XML))
  return spec


def get_goal_cfg(pos: tuple[float, float, float] = (0, 0, 0)) -> EntityCfg:
  """Get goal entity configuration with custom position.

  The goal consists of two vertical posts (y=±1.5, z∈[0,1.8]) and a
  horizontal crossbar (z=1.8, y∈[-1.5, 1.5]). All bodies are static.

  Args:
      pos: World position (x, y, z) offset for the entire goal assembly.

  Returns:
      EntityCfg configured for the goal.
  """
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=pos),
    spec_fn=get_spec,
  )

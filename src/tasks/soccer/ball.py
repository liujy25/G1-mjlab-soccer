"""Ball entity configuration for soccer tasks."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.entity import EntityCfg

BALL_XML: Path = SRC_PATH / "assets" / "soccer" / "ball.xml"
assert BALL_XML.exists()


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(BALL_XML))
  return spec


def get_ball_cfg(pos: tuple[float, float, float] = (0, 0, 0.11)) -> EntityCfg:
  """Get ball entity configuration with custom initial position.

  Args:
      pos: World position (x, y, z) of the ball at reset.

  Returns:
      EntityCfg configured for the ball.
  """
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=pos),
    spec_fn=get_spec,
  )

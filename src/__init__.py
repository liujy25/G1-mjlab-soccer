from pathlib import Path

from src.utils.warp_compat import patch_warp_context_runtime

patch_warp_context_runtime()

SRC_PATH: Path = Path(__file__).parent
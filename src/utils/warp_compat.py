"""Compatibility helpers for Warp API changes."""
from types import SimpleNamespace
from typing import Any


def patch_warp_context_runtime() -> None:
  """Provide the legacy ``wp.context.runtime`` API expected by mjlab 1.2.0.

  Warp 1.13 exposes ``get_cuda_driver_version()`` at the module level and no
  longer publishes ``wp.context``. mjlab 1.2.0 still reads
  ``wp.context.runtime.driver_version`` when deciding whether CUDA graphs are
  available. This shim keeps the behavior local to scripts in this repository
  without modifying the conda environment's site-packages.
  """
  import warp as wp

  context = getattr(wp, "context", None)
  if context is not None and hasattr(context, "runtime"):
    return

  class _RuntimeProxy:
    @property
    def driver_version(self) -> tuple[int, int] | None:
      try:
        return wp.get_cuda_driver_version()
      except Exception as exc:
        print(f"[WARNING] CUDA Graphs disabled: {exc}")
        return None

  runtime: Any = _RuntimeProxy()
  if context is None:
    wp.context = SimpleNamespace(runtime=runtime)
  else:
    context.runtime = runtime

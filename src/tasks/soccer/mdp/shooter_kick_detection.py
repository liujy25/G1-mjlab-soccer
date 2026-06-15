"""Kick contact detection and tracking for coordinated soccer rewards.

Port of HumanoidSoccer's kick_detection.py to mjlab.
Provides KickContactTracker that consolidates ball contact force sensing,
foot identification, and reward window management across multiple reward terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactSensor
  from .shooter_commands import MultiMotionSoccerCommand


@dataclass
class KickContactEvent:
  """Per-step kick contact detection results."""
  new_contact: torch.Tensor   # (num_envs,) bool — first valid contact this episode
  kick_detected: torch.Tensor  # (num_envs,) bool — contact force > threshold
  peak_force: torch.Tensor     # (num_envs,) float


@dataclass
class ContactFootInfo:
  """Resolved foot metadata for environments with an active kick contact."""
  env_ids: torch.Tensor         # (K,) long
  body_indices: torch.Tensor    # (K,) long — which foot body made contact
  sides: torch.Tensor           # (K,) int8 — 0=left, 1=right
  expected: torch.Tensor        # (K,) int8 — expected kick leg from motion


class KickContactTracker:
  """Shared kick contact detection reusable across multiple reward terms.

  Caches detection results per step so all reward terms share one sensor read.
  """

  def __init__(self, env: ManagerBasedRlEnv, state_prefix: str):
    self._env = env
    self._state_prefix = state_prefix
    self._device = env.device
    self._num_envs = env.num_envs
    self._cache_valid = False
    self._cached_event: KickContactEvent | None = None
    self._foot_cache: tuple[torch.Tensor, torch.Tensor] | None = None

  def begin_step(self):
    """Reset per-step cache and handle envs that just resampled motions."""
    self._cache_valid = False
    self._cached_event = None
    flag_name = self._tensor_name("motion_resampled")
    resample_flags = getattr(self._env, flag_name, None)
    if resample_flags is not None and resample_flags.shape[0] == self._num_envs:
      self._handle_resample(resample_flags.to(device=self._device, dtype=torch.bool))

  def detect(
    self,
    command: MultiMotionSoccerCommand,
    ball_sensor_name: str,
    horizontal_force_threshold: float,
  ) -> KickContactEvent:
    """Detect new kick contacts. Evaluated once per step (cached)."""
    if self._cache_valid and self._cached_event is not None:
      return self._cached_event

    ball_sensor = self._get_contact_sensor(ball_sensor_name)
    if ball_sensor is None:
      event = KickContactEvent(
        new_contact=torch.zeros(self._num_envs, dtype=torch.bool, device=self._device),
        kick_detected=torch.zeros(self._num_envs, dtype=torch.bool, device=self._device),
        peak_force=torch.zeros(self._num_envs, dtype=torch.float32, device=self._device),
      )
      self._cached_event = event
      self._cache_valid = True
      return event

    # Prefer substep history so short ball-foot impacts are not missed between
    # policy steps. Fallback keeps sensors without history usable.
    force = getattr(ball_sensor.data, "force_history", None)
    if force is None or force.numel() == 0:
      force = ball_sensor.data.force
    if force is None or force.numel() == 0:
      event = KickContactEvent(
        new_contact=torch.zeros(self._num_envs, dtype=torch.bool, device=self._device),
        kick_detected=torch.zeros(self._num_envs, dtype=torch.bool, device=self._device),
        peak_force=torch.zeros(self._num_envs, dtype=torch.float32, device=self._device),
      )
      self._cached_event = event
      self._cache_valid = True
      return event

    force = force.to(device=self._device)
    # Take max over contact slots and history samples.
    force_norm = torch.linalg.vector_norm(force, dim=-1)
    if force_norm.ndim > 1:
      peak_force = force_norm.amax(dim=tuple(range(1, force_norm.ndim)))
    else:
      peak_force = force_norm
    kick_detected = peak_force > horizontal_force_threshold

    contact_awarded = self._get_or_init_bool("target_contact_awarded", default=False)
    new_contact = (~contact_awarded) & kick_detected
    if torch.any(new_contact):
      contact_awarded[new_contact] = True
    self._update_detection_state(new_contact)

    event = KickContactEvent(new_contact, kick_detected, peak_force)
    self._cached_event = event
    self._cache_valid = True
    return event

  def record_expected_success(self, mask: torch.Tensor, expected_mask: torch.Tensor):
    """Store whether a detected kick matched the expected leg."""
    state = self._get_or_init_bool("expected_kick_success", default=False)
    state[mask] = expected_mask[mask]

  def get_contact_awarded(self) -> torch.Tensor:
    return self._get_or_init_bool("target_contact_awarded", default=False)

  def freeze_proximity_reward(self, env_ids: torch.Tensor, values: torch.Tensor):
    frozen = self._get_or_init_float("frozen_proximity_reward", default=0.0)
    frozen[env_ids] = values

  def get_frozen_proximity_reward(self) -> torch.Tensor:
    return self._get_or_init_float("frozen_proximity_reward", default=0.0)

  def resolve_contact_foot(
    self,
    command: MultiMotionSoccerCommand,
    foot_cfg: SceneEntityCfg,
    mask: torch.Tensor,
  ) -> ContactFootInfo:
    """Determine which foot made contact for each env."""
    env_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if env_ids.numel() == 0:
      empty = torch.zeros(0, dtype=torch.long, device=self._device)
      zeros_i8 = torch.zeros(0, dtype=torch.int8, device=self._device)
      return ContactFootInfo(empty, empty, zeros_i8, zeros_i8)

    body_indices, sides = self._get_foot_metadata(command, foot_cfg)
    robot = command.robot

    foot_pos = robot.data.body_link_pos_w[env_ids][:, body_indices]
    ball_pos = command.soccer_ball_pos[env_ids]
    env_origins = self._env.scene.env_origins
    if env_origins is not None:
      ball_pos = ball_pos + env_origins[env_ids]

    diff = torch.norm(foot_pos - ball_pos.unsqueeze(1), dim=-1)
    closest_idx = torch.argmin(diff, dim=-1)
    selected_body_indices = body_indices[closest_idx]
    hit_sides = sides[closest_idx]

    expected = command.kick_leg[env_ids].to(torch.int8).clamp(min=0)
    return ContactFootInfo(env_ids, selected_body_indices, hit_sides, expected)

  # -- Internal helpers ---------------------------------------------------------

  def _get_contact_sensor(self, name: str):
    sensors = self._env.scene.sensors
    if sensors is None:
      return None
    if isinstance(sensors, dict):
      return sensors.get(name)
    try:
      return sensors[name]
    except (KeyError, TypeError):
      return None

  def _tensor_name(self, suffix: str) -> str:
    return f"{self._state_prefix}_{suffix}"

  def _get_or_init_bool(self, suffix: str, default: bool) -> torch.Tensor:
    name = self._tensor_name(suffix)
    t = getattr(self._env, name, None)
    if t is None or t.shape[0] != self._num_envs:
      t = torch.full((self._num_envs,), default, dtype=torch.bool, device=self._device)
      setattr(self._env, name, t)
    return t.to(device=self._device, dtype=torch.bool)

  def _get_or_init_float(self, suffix: str, default: float) -> torch.Tensor:
    name = self._tensor_name(suffix)
    t = getattr(self._env, name, None)
    if t is None or t.shape[0] != self._num_envs:
      t = torch.full((self._num_envs,), default, dtype=torch.float32, device=self._device)
      setattr(self._env, name, t)
    return t.to(device=self._device, dtype=torch.float32)

  def _update_detection_state(self, new_contact: torch.Tensor):
    if not torch.any(new_contact):
      return
    s = self._get_or_init_bool("kick_success", default=False)
    s[new_contact] = True

  def _get_foot_metadata(
    self,
    command: MultiMotionSoccerCommand,
    foot_cfg: SceneEntityCfg,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if self._foot_cache is not None:
      return self._foot_cache
    robot = self._env.scene[foot_cfg.name]
    indices = torch.as_tensor(
      robot.find_bodies(foot_cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self._device,
    )
    sides = torch.tensor(
      [0 if "left" in name.lower() else 1 if "right" in name.lower() else -1
       for name in foot_cfg.body_names],
      dtype=torch.int8,
      device=self._device,
    )
    self._foot_cache = (indices, sides)
    return self._foot_cache

  def _handle_resample(self, resample_flags: torch.Tensor):
    """Reset kick state for environments that just resampled motions."""
    if not torch.any(resample_flags):
      return
    contact = self._get_or_init_bool("target_contact_awarded", default=False)
    success = self._get_or_init_bool("kick_success", default=False)
    expected = self._get_or_init_bool("expected_kick_success", default=False)
    frozen = self._get_or_init_float("frozen_proximity_reward", default=0.0)

    contact[resample_flags] = False
    success[resample_flags] = False
    expected[resample_flags] = False
    frozen[resample_flags] = 0.0

    # Reset reward timers/state to avoid stale sparse reward windows.
    for suffix in [
      "dir_align_timer",
      "dir_align_prev",
      "speed_timer",
      "speed_prev",
      "z_speed_timer",
      "z_speed_prev",
    ]:
      name = self._tensor_name(suffix)
      t = getattr(self._env, name, None)
      if t is not None and t.shape[0] == self._num_envs:
        t[resample_flags] = 0

    resample_flags[resample_flags] = False
    setattr(self._env, self._tensor_name("motion_resampled"), resample_flags)

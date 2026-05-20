"""Motion dataset loading and per-episode playback for soccer kick motions.

Loads retargeted .npz motion files (29-DoF G1 joints) and provides
per-frame reference trajectories used as O^ref_t in the observation space,
matching the HumanoidSoccer paper's motion tracking framework.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import torch


def _pad_tensor_list_3d(tensor_list: list[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
  """Pad a list of 2-D or 3-D tensors to the same T length and stack into a 4-D tensor.

  Returns tensor of shape (num_files, max_T, ...).
  """
  max_T = max(t.shape[0] for t in tensor_list)
  padded = []
  for t in tensor_list:
    if t.shape[0] < max_T:
      pad_shape = (max_T - t.shape[0],) + t.shape[1:]
      pad = torch.full(pad_shape, pad_value, dtype=t.dtype, device=t.device)
      t = torch.cat([t, pad], dim=0)
    padded.append(t)
  return torch.stack(padded, dim=0)


@dataclass
class MotionInfo:
  """Metadata for a single motion file."""
  name: str
  num_frames: int
  kick_leg: str  # "left", "right", or "unknown"
  fps: int


class MotionDataset:
  """Load and manage a collection of retargeted soccer kick motion .npz files.

  Each .npz file contains per-frame reference data for a single kick motion:
    - joint_pos:     (T, 29)     float32  reference joint positions (rad)
    - joint_vel:     (T, 29)     float32  reference joint velocities (rad/s)
    - body_pos_w:    (T, 30, 3)  float32  world-space body positions
    - body_quat_w:   (T, 30, 4)  float32  world-space body orientations (wxyz)
    - body_lin_vel_w:(T, 30, 3)  float32
    - body_ang_vel_w:(T, 30, 3)  float32  body angular velocities
    - fps:           scalar      int64    sampling rate (50 Hz)
    - kick_leg:      str         "left" / "right"

  All motions are padded to the same length (max_T across files).
  """

  # G1 MJCF body indices (30 bodies, consistent between Isaac Lab USD and MuJoCo MJCF).
  # Key bodies used for termination conditions and observations.
  BODY_TORSO = 15
  BODY_LEFT_ANKLE_ROLL = 6
  BODY_RIGHT_ANKLE_ROLL = 12
  BODY_LEFT_WRIST_YAW = 22
  BODY_RIGHT_WRIST_YAW = 29

  def __init__(self, motion_dir: str, device: str = "cpu"):
    self.device = device
    self.motion_dir = motion_dir

    files = sorted(glob.glob(os.path.join(motion_dir, "*.npz")))
    if not files:
      raise FileNotFoundError(f"No .npz motion files found in {motion_dir}")

    self.infos: list[MotionInfo] = []
    joint_pos_list: list[torch.Tensor] = []
    joint_vel_list: list[torch.Tensor] = []
    body_pos_w_list: list[torch.Tensor] = []
    body_quat_w_list: list[torch.Tensor] = []
    body_ang_vel_list: list[torch.Tensor] = []
    self._file_lengths: list[int] = []

    for f in files:
      data = np.load(f)
      name = os.path.splitext(os.path.basename(f))[0]
      n_frames = int(data["joint_pos"].shape[0])
      fps = int(data["fps"].item()) if "fps" in data else 50

      kick_leg = "unknown"
      if "kick_leg" in data.files:
        raw = str(data["kick_leg"]).strip().lower()
        if raw in ("left", "right"):
          kick_leg = raw

      self.infos.append(MotionInfo(name=name, num_frames=n_frames, kick_leg=kick_leg, fps=fps))
      self._file_lengths.append(n_frames)

      joint_pos_list.append(torch.tensor(data["joint_pos"], dtype=torch.float32, device=device))
      joint_vel_list.append(torch.tensor(data["joint_vel"], dtype=torch.float32, device=device))
      body_pos_w_list.append(torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device))
      body_quat_w_list.append(torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device))
      body_ang_vel_list.append(torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device))

    self.num_motions = len(files)
    self.file_lengths = torch.tensor(self._file_lengths, dtype=torch.long, device=device)
    self.max_frames = max(self._file_lengths)

    self.joint_pos = _pad_tensor_list_3d(joint_pos_list)        # (M, max_T, 29)
    self.joint_vel = _pad_tensor_list_3d(joint_vel_list)        # (M, max_T, 29)
    self.body_pos_w = _pad_tensor_list_3d(body_pos_w_list)      # (M, max_T, 30, 3)
    self.body_quat_w = _pad_tensor_list_3d(body_quat_w_list)    # (M, max_T, 30, 4)
    self.body_ang_vel_w = _pad_tensor_list_3d(body_ang_vel_list)  # (M, max_T, 30, 3)

    self.joint_dim = self.joint_pos.shape[-1]  # 29

  def _clamp(self, motion_idx: torch.Tensor, time_steps: torch.Tensor):
    """Return clamped (m, t) indices safe for indexing."""
    t = torch.clamp(time_steps, 0, self.max_frames - 1)
    m = torch.clamp(motion_idx, 0, self.num_motions - 1)
    return m, t

  def get_frame(self, motion_idx: torch.Tensor, time_steps: torch.Tensor):
    """Return (joint_pos, joint_vel, anchor_ang_vel) for current frame."""
    m, t = self._clamp(motion_idx, time_steps)
    jp = self.joint_pos[m, t]
    jv = self.joint_vel[m, t]
    bav = self.body_ang_vel_w[m, t, self.BODY_TORSO, :]
    return jp, jv, bav

  def get_ref_body_pos(self, motion_idx: torch.Tensor, time_steps: torch.Tensor,
                        body_index: int) -> torch.Tensor:
    """Reference world-frame position of a specific body at current frame.

    Returns (num_envs, 3).  Positions are stored WITHOUT env_origins;
    callers that need world-frame coordinates must add env_origins themselves.
    """
    m, t = self._clamp(motion_idx, time_steps)
    return self.body_pos_w[m, t, body_index, :]

  def get_ref_body_quat(self, motion_idx: torch.Tensor, time_steps: torch.Tensor,
                        body_index: int) -> torch.Tensor:
    """Reference world-frame orientation (wxyz) of a specific body at current frame.

    Returns (num_envs, 4).
    """
    m, t = self._clamp(motion_idx, time_steps)
    return self.body_quat_w[m, t, body_index, :]

  def get_motion_kick_leg(self, motion_idx: int) -> str:
    return self.infos[motion_idx].kick_leg


class MotionCommand:
  """Per-environment motion playback state, attached to the env.

  Provides per-frame reference data for observations (O^ref_t) and
  termination conditions (anchor / end-effector deviation checks).
  """

  def __init__(self, dataset: MotionDataset, num_envs: int = 1, device: str = "cpu"):
    self.dataset = dataset
    self.num_envs = num_envs
    self.device = device

    self.motion_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.time_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.motion_length = torch.zeros(num_envs, dtype=torch.long, device=device)

    self._init_random()

  def _init_random(self):
    if self.dataset.num_motions > 1:
      self.motion_idx = torch.randint(0, self.dataset.num_motions, (self.num_envs,),
                                       dtype=torch.long, device=self.device)
    self.motion_length = self.dataset.file_lengths[self.motion_idx]

  def reset(self):
    self._init_random()
    self.time_steps.zero_()

  def step(self) -> torch.Tensor:
    """Advance one frame. Returns mask of envs that just resampled."""
    self.time_steps += 1
    done = self.time_steps >= self.motion_length
    if done.any():
      env_ids = done.nonzero(as_tuple=True)[0]
      self._resample(env_ids)
    return done

  def _resample(self, env_ids: torch.Tensor):
    n = len(env_ids)
    if self.dataset.num_motions > 1:
      self.motion_idx[env_ids] = torch.randint(0, self.dataset.num_motions, (n,),
                                                dtype=torch.long, device=self.device)
    self.motion_length[env_ids] = self.dataset.file_lengths[self.motion_idx[env_ids]]
    self.time_steps[env_ids] = 0

  # -- Observation-facing properties -------------------------------------------

  @property
  def joint_pos_ref(self) -> torch.Tensor:
    """(num_envs, 29) reference joint positions."""
    jp, _, _ = self.dataset.get_frame(self.motion_idx, self.time_steps)
    return jp

  @property
  def joint_vel_ref(self) -> torch.Tensor:
    """(num_envs, 29) reference joint velocities."""
    _, jv, _ = self.dataset.get_frame(self.motion_idx, self.time_steps)
    return jv

  @property
  def anchor_ang_vel_ref(self) -> torch.Tensor:
    """(num_envs, 3) reference anchor angular velocity."""
    _, _, bav = self.dataset.get_frame(self.motion_idx, self.time_steps)
    return bav

  # -- Termination-facing properties -------------------------------------------

  @property
  def anchor_pos_w_ref(self) -> torch.Tensor:
    """(num_envs, 3) reference torso (anchor) position in world frame (no env_origins)."""
    return self.dataset.get_ref_body_pos(self.motion_idx, self.time_steps,
                                         MotionDataset.BODY_TORSO)

  @property
  def anchor_quat_w_ref(self) -> torch.Tensor:
    """(num_envs, 4) reference torso orientation in world frame (wxyz)."""
    return self.dataset.get_ref_body_quat(self.motion_idx, self.time_steps,
                                          MotionDataset.BODY_TORSO)

  def get_ee_pos_w_ref(self, body_index: int) -> torch.Tensor:
    """(num_envs, 3) reference end-effector position in world frame (no env_origins)."""
    return self.dataset.get_ref_body_pos(self.motion_idx, self.time_steps, body_index)

# CS2810 — Humanoid Robot Soccer

A reinforcement learning template for humanoid robot soccer (shooting + goalkeeping)
with the Unitree G1 on MuJoCo physics. Built on the
[unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) framework. More instructions can be found [here](./doc/instruction/cs2810-proj.pdf).

## Overview

- **Task**: G1 humanoid shoots a stationary ball toward a goal (shooter) and intercepts incoming balls (goalkeeper)
- **Physics**: MuJoCo with fluid air-drag on the ball, 50 Hz control
- **Robot**: Unitree G1, 29-DoF PD position control
- **Template**: Simplified from [HumanoidSoccer](https://github.com/TeleHuman/HumanoidSoccer) and [Humanoid-Goalkeeper](https://github.com/InternRobotics/Humanoid-Goalkeeper) reference implementations

## Setup

See [doc/setup_en.md](doc/setup_en.md) for environment installation.

## Quick Start

```bash
# List all tasks
python scripts/list_envs.py

# --- Shooter ---

# Play (visualize with zero agent)
python scripts/play.py Unitree-G1-Shooter --agent zero --viewer native

# Eval shooter
python scripts/eval_naive_shooter.py --headless --num-trials 50
python scripts/eval_naive_shooter.py --headless --num-trials 50 --checkpoint <path>

# --- Goalkeeper ---

# Play (visualize with zero agent)
python scripts/play.py Unitree-G1-Goalkeeper --agent zero --viewer native

# Eval goalkeeper
python scripts/eval_naive_goalkeeper.py --headless --num-trials 50
python scripts/eval_naive_goalkeeper.py --headless --num-trials 50 --checkpoint <path>
```

## Evaluation

### Shooter

Scene: G1 near origin facing -y (motion-local coords, identical to training/play). Goal placed at (0, -5, 0) rotated 90° to face G1, ball placed dynamically by the command system. No `motion_origin_offset` / `motion_yaw_offset` transform — eval uses the exact same coordinate system as training and play mode.

**Metrics** (matching HumanoidSoccer §IV-B):
- **Success Rate** — fraction of episodes where ball crosses goal plane (y≤-5, |x|≤1.5m, z≤1.8m)
- **Kick Accuracy** — cosine similarity between ball velocity direction and ball→goal-center vector
- **Kick Speed** — ball speed when first > 1 m/s

### Goalkeeper

Scene: G1 at goal line (0, 0, 0.8), yaw=0 faces +x. Ball launched via 6-region parabolic trajectory model from +x (3-5m front) toward -x (behind). Goal at (-0.5, 0, 0) behind G1.

**Ball trajectory** (matching Humanoid-Goalkeeper §III-A):
- 6 landing regions: Right/Left × Mid/Up/Low
- Ball start: +x 3–5m in front, random y/z within region bounds
- Ball end: -x 0.1–0.6m behind robot, within sampled region
- Flight time: 0.6–1.0s
- Parabolic velocity: v_xy = Δxy / t, v_z = (Δz + ½gt²) / t

**Metrics** (matching Humanoid-Goalkeeper §IV):
- **Block Rate (Esucc)** — fraction of episodes where ball velocity drops > 2 m/s when behind robot
- **Min ball-robot xy distance** — closest approach during the episode
- **Mean ball speed at robot** — ball speed when crossing the goal line

## Project Structure

```
src/
  assets/soccer/
    ball.xml, goal.xml, ground.xml     # MuJoCo entity models
    motions/
      shooter/                         # Retargeted kick trajectories (13 .npz)
      goalkeeper/                      # GK human motion data (6 .pt + joint_id.txt)
    weight/
      goalkeeper.pt                    # Reference pretrained HIMPPO checkpoint
  tasks/soccer/
    ball.py, goal.py, ground.py        # Entity config factories
    soccer_env_cfg.py                  # Base env-cfg factory
    motion_data.py                     # MotionDataset + MotionCommand (legacy)
    modules/
      gk_actor_critic.py               # GoalkeeperActorCritic — HIMPPO architecture replica
    mdp/
      shared_obs.py                    # Shared observation functions
      shared_rewards.py                # Shared reward functions (is_terminated)
      shared_terminations.py           # Shared termination functions (timeout, bad_orientation)
      shared_reset.py                  # Shared reset functions (root state, joints)
      shared_domain_randomization.py   # Shared DR (push_robot, perturb_ball)
      shooter_commands.py              # MultiMotionSoccerCommand — motion playback + ball placement
      shooter_kick_detection.py        # KickContactTracker — shared contact detection
      shooter_rewards.py               # Shooter kick reward functions (9 funcs)
      shooter_obs.py                   # Shooter privileged critic + perception obs
      goalkeeper_rewards.py            # GK reward functions (7 funcs + state reset)
      goalkeeper_obs.py                # GK privileged critic obs + PD gain configs
      goalkeeper_ball_reset.py         # 6-region parabolic ball trajectory reset
    config/
      settings.yaml                    # Central parameter source of truth
      soccer_settings.py               # Typed settings loader (dataclass-backed)
      g1/
        env_cfgs.py                    # Shooter & goalkeeper environment configs
        rl_cfg.py                      # PPO config + GoalkeeperRunner
        training_env_cfgs.py           # Training config factories (Stage I/II + GK) — reference
      eval/
        eval_shooter_cfg.py            # Shooter eval (reuses Stage II play config + goal)
        eval_goalkeeper_cfg.py         # Goalkeeper eval (T=10 history, 960D/113D)
      training/                        # Reference training configs (unregistered)
        stage1_env_cfg.py              # Stage I: motion tracking, adaptive sampling
        stage2_env_cfg.py              # Stage II: perception-guided kicking
        goalkeeper_env_cfg.py          # GK training: single-stage reactive
scripts/
  train.py                             # Training entrypoint
  play.py                              # Interactive visualization
  eval_naive_shooter.py                # Shooter eval (headless stats or viewer)
  eval_naive_goalkeeper.py             # Goalkeeper eval (headless stats or viewer)
```

## For CS2810 Students

This repository is a **simplified template** based on the reference implementations of [HumanoidSoccer](https://arxiv.org/abs/2602.05310) and [Humanoid-Goalkeeper](https://github.com/InternRobotics/Humanoid-Goalkeeper).
It provides:

- **Playable environments**: `Unitree-G1-Shooter` and `Unitree-G1-Goalkeeper` with
  configurable scene layout, ball physics, and MuJoCo visualization.
- **Eval scripts**: `eval_naive_shooter.py` and `eval_naive_goalkeeper.py` that load
  checkpoints, run headless batch trials, collect paper metrics, and record videos.
- **Reference training configs**: The `config/training/` directory contains the
  complete training environment designs (observations, rewards, terminations,
  domain randomization) from both papers. These are provided as design reference
  only — they are **not registered as tasks**.

**You need to implement training yourself.** (*Jinxi's Note: actually you cannot get full 60% credit by just running these configs, you need to understand the design and implement your own training pipeline.*)

> **Note**: Adversarial play (loading two G1 robots simultaneously in the same scene)
> is **not yet implemented**. This feature will be added in a future update to support
> Phase 2 of the project (the adversarial tournament).

## Settings (`config/settings.yaml`)

```yaml
ball:              # radius=0.10, mass=0.35
goal:              # width=3.0, height=1.8
penalty_spot:      # distance_from_goal=4.0
scene:
  goal_pos: [0,0,0]
  goalkeeper_pos: [0, 0, 0.8]
  shooter_behind_ball: 1.0
goalkeeper_regions:  # 6 regions (height z × width y)
goalkeeper_training:
  ee_reach_std: 0.3
  stop_ball_vel_drop: 2.0
  behind_robot_x: 0.0
ball_trajectory:
  ball_start_distance: [3.0, 5.0]
  ball_end_distance: [0.1, 0.6]
  t_flight: [0.5, 1.0]
episode_length_s: 10.0                 # shooter
goalkeeper_episode_length_s: 3.0       # goalkeeper
```

## Acknowledgements

Built for CS2810 (Spring 2026). This project uses motion data and design references from [HumanoidSoccer](https://github.com/TeleHuman/HumanoidSoccer) and [Humanoid-Goalkeeper](https://github.com/InternRobotics/Humanoid-Goalkeeper). If you use this template, please cite:

```
@article{ren2025humanoidgoalkeeper,
  title={Humanoid Goalkeeper: Learning from Position Conditioned Task-Motion Constraints},
  author={Ren, Junli, Long, Jungfeng, Huang, Tao and Wang, Huayi, Wang, Zirui and Jia, Feiyu, Zhang, Wentao and Wang, Jingbo, Ping Luo and Pang, Jiangmiao},
  year={2025}
}
@misc{kong2026learningsoccerskillshumanoid,
  title={Learning Soccer Skills for Humanoid Robots: A Progressive Perception-Action Framework},
  author={Jipeng Kong and Xinzhe Liu and Yuhang Lin and Jinrui Han and Sören Schwertfeger and Chenjia Bai and Xuelong Li},
  year={2026},
  eprint={2602.05310},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2602.05310}
}
```

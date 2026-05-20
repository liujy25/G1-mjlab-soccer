# CS 2810 — Humanoid Robot Soccer

A perception-guided humanoid soccer shooting and intercepting project for Unitree G1, using
reinforcement learning with motion tracking on MuJoCo physics.  Built on the
[unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) framework.

## Overview

- **Task**: G1 humanoid shoots a stationary ball or intercept the ball
- **Physics**: MuJoCo with fluid air-drag on the ball, 50 Hz control
- **Robot**: Unitree G1, 29-DoF PD position control, armature-based stiffness

## Setup

See [doc/setup_en.md](doc/setup_en.md) for environment installation.

## Quick Start

```bash
# List all registered tasks
python scripts/list_envs.py

# Visualize the shooter scene (zero-agent, robot holds default pose)
python scripts/play.py Unitree-G1-Naive-Shooter --agent=zero

# Headless batch evaluation with metrics
python scripts/eval_naive_shooter.py --headless --num-trials=50

# Load a trained checkpoint
python scripts/eval_naive_shooter.py \
    --headless \
    --num-trials=50 \
    --checkpoint logs/rsl_rl/g1_soccer/model_5000.pt

# Record a video
python scripts/eval_naive_shooter.py --video --video-length=500
```

## Observation Space (Eval)

```
 o_t = (o^prop_t,  o^ref_t,  o^soc_t)

 o^prop : proprioception   — projected_gravity (3) + base_ang_vel (3)
                            + joint_pos (29) + joint_vel (29) + last_action (29)   =  93 D
 o^ref  : motion reference — ref_joint_pos (29) + ref_joint_vel (29)
                            + ref_anchor_ang_vel (3)                               =  61 D
 o^soc  : soccer perception — ball_in_robot_frame (3) + goal_in_robot_frame (3)   =   6 D
 ─────────────────────────────────────────────────────────────────────────────────────────
 Actor total                                                                        160 D
 Critic total (actor + base_lin_vel)                                                163 D
```

## Evaluation Protocol

Ball position is fixed at the penalty spot (4 m from goal).  The robot root pose
is randomized each episode (±0.5 m xy, ±0.5 rad yaw).

**Metrics** (matching HumanoidSoccer Section IV‑B):
- Success Rate — fraction of episodes where the ball enters the goal
- Kick Accuracy — cosine similarity between ball velocity direction and the
  ball-to-goal-center vector

```bash
# Interactive viewer
python scripts/eval_naive_shooter.py

# Batch evaluation (N trials, no viewer)
python scripts/eval_naive_shooter.py --headless --num-trials=100

# With a trained policy
python scripts/eval_naive_shooter.py --headless --num-trials=100 --checkpoint <path>
```

## Project Structure

```
src/
  assets/soccer/
    ball.xml, goal.xml, ground.xml     # MuJoCo entity models
    motions/                           # Retargeted kick trajectories (.npz)
  tasks/soccer/
    ball.py, goal.py, ground.py        # Entity config factories
    soccer_env_cfg.py                  # Base env-cfg factory
    motion_data.py                     # Motion dataset loading & playback
    mdp/__init__.py                    # Observation, reward, termination, DR, reset functions
    config/
      settings.yaml                    # Single source of truth for soccer parameters
      soccer_settings.py               # Typed settings loader (dataclass-backed)
      g1/env_cfgs.py                   # G1 shooter & goalkeeper training configs
      g1/rl_cfg.py                     # PPO config
      eval/eval_shooter_cfg.py         # Shooter eval config (paper observation space + DR)
      eval/eval_goalkeeper_cfg.py      # Goalkeeper eval config
scripts/
  play.py                              # Task-agnostic scene viewer
  eval_naive_shooter.py                # Shooter eval (headless stats or interactive viewer)
  eval_naive_goalkeeper.py             # Goalkeeper eval
```

## Acknowledgements

Built for CS 2810 (Spring 2026).  This project uses motion data and design references from the [HumanoidSoccer](https://github.com/TeleHuman/HumanoidSoccer) and [Humanoid-Goalkeeper] repositories. If you use this template, please site 

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

# Unitree RL Mjlab — Soccer

A humanoid robot soccer RL project built on [mjlab](https://github.com/mujocolab/mjlab.git),
using MuJoCo as the physics backend. Currently supports Unitree G1 Shooter and Goalkeeper tasks.


## Setup

Please refer to [setup_en.md](doc/setup_en.md) for installation and configuration.


## List Environments

List all registered task environments:

```bash
python scripts/list_envs.py
```

Currently registered soccer environments:
- `Unitree-G1-Naive-Shooter` — G1 at penalty spot facing goal, ball placed ahead
- `Unitree-G1-Naive-Goalkeeper` — G1 at goal line facing the ball


## Visualize the Scene

Use `play.py` with `--agent=zero` to inspect the scene layout (no policy loaded, robot holds default pose):

```bash
# View Shooter scene
python scripts/play.py Unitree-G1-Naive-Shooter --agent=zero

# View Goalkeeper scene
python scripts/play.py Unitree-G1-Naive-Goalkeeper --agent=zero
```

> `--agent=zero` outputs zero actions, so the robot stays in its default standing pose — useful for checking the relative positions of ball, goal, and robot.


## Scene Layout

```
        Goal (x=0, 3.0m × 1.8m)
    +----|          |----+
    |    |          |    |
    |    |    GK    |    |     ← goalkeeper G1 at (0, 0, 0.8), facing -x
    |    |          |    |
    |             |
    |    Ball     |            ← ball at (-6.0, 0, 0.11)
    |     G1      |            ← shooter G1 at (-6.2, 0, 0.8), facing +x
    +-------------+
```

- Ball is 6m from the goal line
- Goalkeeper: ball launched at 1 m/s toward goal center (simulates a weak shot)


## Acknowledgements

- [mjlab](https://github.com/mujocolab/mjlab.git) — training and execution framework
- [Humanoid-Goalkeeper](https://github.com/InternRobotics/Humanoid-Goalkeeper) — goalkeeper design reference
- [HumanoidSoccer](https://github.com/TeleHuman/HumanoidSoccer) — shooter design reference

# ros2_lerobot

ROS2 workspace for LeRobot integration, teleoperation, reinforcement learning, and simulation.

## Project Overview

This is a ROS2 workspace that bridges LeRobot with ROS2 ecosystems, enabling:
- **Teleoperation** of robot arms (SO101 leader/follower, etc.)
- **Reinforcement Learning** workflows via LeRobot
- **Simulation** in Gazebo, Genesis, or Isaac Sim

## Tech Stack

- **ROS2**: Jazzy (via RoboStack)
- **Package Manager**: Pixi
- **ML Framework**: LeRobot (with Feetech support)
- **Python**: 3.12
- **Simulators**: Gazebo, Genesis, Isaac Sim (planned)

## Workspace Structure

```
ros2_lerobot/
├── src/
│   ├── lerobot_robots_robots/             # Follower robot bridge (SO101, etc.)
│   ├── lerobot_robots_teleoperators/      # Leader teleoperator bridge (SO101, etc.)
│   ├── lerobot_robots_description/        # URDF/Xacro robot descriptions (combined)
│   ├── lerobot_robots_control/            # ros2_control configs (combined)
│   └── lerobot_robots_bringup/            # Launch files & runtime config (combined)
├── config/                                # Global config YAML files
├── pixi.toml                              # Pixi workspace config
├── docker-compose.yml                     # Docker environment
└── Dockerfile
```

## Package Organization

### `lerobot_robots_description`
- URDF/Xacro files for all robot models
- Mesh files (STL, DAE)
- Robot visual & collision geometry
- TF tree definitions

### `lerobot_robots_control`
- Controller configurations (YAML)
- Hardware interface plugins
- Joint state publishers
- ros2_control definitions

### `lerobot_robots_bringup`
- Launch files for real robot & simulation
- RViz configurations
- Node lifecycle management
- Parameter files

### `lerobot_robots_teleoperators`
- Bridge between LeRobot leader arms and ROS2
- `lerobot_teleoperator_node` - publishes leader joint states to `/joint_states`
- Config-driven via `--config path/to/teleop_config.yaml`

### `lerobot_robots_robots`
- Bridge between LeRobot follower arms and ROS2
- `lerobot_robot_node` - subscribes to `/joint_commands`, sends actions to follower
- Config-driven via `--config path/to/robot_config.yaml`

## Common Commands

### Build
```bash
# Full build
pixi run build2

# Selective build (robots + teleoperators)
pixi run build
```

### Run Nodes
```bash
# Run leader (SO101 teleoperator)
pixi run run-so101-leader
# Equivalent to: ros2 run lerobot_robots_teleoperators lerobot_teleoperator_node --config config/teleop_so101.yaml

# Run follower (SO101 robot)
pixi run run-so101-follower
# Equivalent to: ros2 run lerobot_robots_robots lerobot_robot_node --config config/robot_so101.yaml
```

### Launch
```bash
# SO101 leader via launch file (uses default config)
ros2 launch lerobot_robots_bringup so101_leader.launch.py

# With custom config
ros2 launch lerobot_robots_bringup so101_leader.launch.py config:=/path/to/custom.yaml
```

### Visualization
```bash
rqt_graph
ros2 run rqt_tf_tree rqt_tf_tree --force-discover
rviz2
```

## Config File Format

### Teleoperator config (`config/teleop_so101.yaml`)
```yaml
port: /dev/ttyACM0
id: leader
use_degrees: true
```

### Robot config (`config/robot_so101.yaml`)
```yaml
port: /dev/ttyACM1
id: follower
use_degrees: true
disable_torque_on_disconnect: true
```

## LeRobot Commands

### Calibration
```bash
# Calibrate follower arm
python3 -m lerobot.calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower

# Calibrate leader arm
python3 -m lerobot.calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader
```

### Teleoperation
```bash
python3 -m lerobot.teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader
```

## Docker

```bash
# Setup display access (host)
xhost +

# Build and start
docker compose up -d --build
docker exec -it ros2_lerobot bash
```

## Development Conventions

- Keep packages focused: description, control, bringup
- Use `.launch.py` format for launch files
- Parameter files in `config/` directory within each package
- URDF files in `urdf/`, meshes in `meshes/`
- Follow ROS2 REP-140 naming conventions (snake_case for packages)
- Nodes use `--config` arg for YAML config (LeRobot-style)

## Future Work

- [ ] Add Gazebo simulation support (ros_gz)
- [ ] Add Genesis simulator integration
- [ ] Add Isaac Sim / Omniverse support
- [ ] RL training pipeline with LeRobot datasets
- [ ] RViz configuration panels for teleoperation
- [ ] Add support for more robot types (SO100, Koch, etc.)

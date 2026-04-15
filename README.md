# ros2_lerobot

ROS2 workspace for LeRobot integration, teleoperation, reinforcement learning, and simulation.

## Project Overview

This is a ROS2 workspace that bridges LeRobot with ROS2 ecosystems, enabling:
- **Teleoperation** of robot arms (SO101 leader/follower, etc.)
- **Reinforcement Learning** workflows via LeRobot
- **Simulation** in Gazebo, Genesis, or Isaac Sim (planned)

## Tech Stack

- **ROS2**: Jazzy (via RoboStack)
- **Package Manager**: Pixi
- **ML Framework**: LeRobot (with Feetech support)
- **Python**: 3.12
- **Simulators**: Gazebo, Genesis, Isaac Sim (planned)

## Setup & Build

This project uses [Pixi](https://pixi.sh) for dependency management and environment setup.

```bash
# Install dependencies
pixi install

# Build the workspace
pixi run build
```

## Teleoperation (SO101)

It is recommended to use the provided **pixi tasks** for running the teleoperation nodes. The nodes are namespaced to avoid topic conflicts.

### 1. Run Leader (Teleoperator)
```bash
pixi run run-so101-leader
```
*Publishes to: `/leader/joint_states`*

### 2. Run Follower (Robot)
```bash
pixi run run-so101-follower
```
*Subscribes to: `/follower/joint_command`*

### 3. Run Leader-Follower Bridge
This node relays the leader's joint states to the follower's joint commands.
```bash
pixi run run-leader-follower
```

## Visualization

### View Follower in RViz
To visualize the robot model and its current state (TFs) in RViz:
```bash
pixi run view-follower
```
*Note: In RViz, set the **Fixed Frame** to `base` and add a **RobotModel** with Description Topic `/follower/robot_description`.*

### Other Tools
```bash
pixi run ros2 run rqt_graph rqt_graph
pixi run ros2 run rqt_tf_tree rqt_tf_tree --force-discover
```

## Workspace Structure

```
ros2_lerobot/
├── src/
│   ├── lerobot_robots_robots/             # Follower robot bridge
│   ├── lerobot_robots_teleoperators/      # Leader teleoperator bridge + bridge node
│   ├── lerobot_robots_description/        # URDF/Xacro robot descriptions
│   ├── lerobot_robots_control/            # ros2_control configurations
│   └── lerobot_robots_bringup/            # Launch files & runtime config
├── config/                                # Global config YAML files
└── pixi.toml                              # Pixi workspace config
```

## Configuration

Configuration files are located in the `config/` directory.

### Teleoperator config (`config/teleop_so101.yaml`)
```yaml
type: so101_leader
port: /dev/ttyACM0
id: leader
use_degrees: true
```

### Robot config (`config/robot_so101.yaml`)
```yaml
type: so101_follower
port: /dev/ttyACM1
id: follower
use_degrees: true
disable_torque_on_disconnect: true
```

## Development Conventions

- Keep packages focused: description, control, bringup.
- Use `.launch.py` format for launch files.
- Use namespaces (`/leader`, `/follower`) for multi-robot setups.
- Use Pixi tasks for common workflows.

## Future Work

- [ ] Add Gazebo simulation support (ros_gz)
- [ ] Add Genesis simulator integration
- [ ] Add Isaac Sim / Omniverse support
- [ ] RL training pipeline with LeRobot datasets

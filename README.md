# ros2_lerobot

ROS2 workspace for LeRobot integration, teleoperation, reinforcement learning, and simulation.

## Quick Start

### Docker Setup
```bash
xhost +
docker compose up -d --build
docker exec -it ros2_lerobot bash
```

### Build
```bash
# Full build
pixi run build2

# Selective build (robots + teleoperators)
pixi run build
```

### Run Nodes

#### Leader (SO101 Teleoperator)
```bash
pixi run run-so101-leader
# Or via launch file
ros2 launch lerobot_robots_bringup so101_leader.launch.py
ros2 launch lerobot_robots_bringup teleoperator.launch.py config:=/path/to/custom.yaml
```

#### Follower (SO101 Robot)
```bash
pixi run run-so101-follower
# Or via launch file
ros2 launch lerobot_robots_bringup so101_follower.launch.py
ros2 launch lerobot_robots_bringup robot.launch.py config:=/path/to/custom.yaml
```

## Workspace Structure

```
ros2_lerobot/
├── src/
│   ├── lerobot_robots_robots/             # Follower robot bridge
│   ├── lerobot_robots_teleoperators/      # Leader teleoperator bridge
│   ├── lerobot_robots_description/        # URDF/Xacro (combined)
│   ├── lerobot_robots_control/            # ros2_control (combined)
│   └── lerobot_robots_bringup/            # Launch files & configs
├── config/                                # Global config YAMLs
└── pixi.toml
```

## Config Files

### Teleoperator (`config/teleop_so101.yaml`)
```yaml
type: so101_leader
port: /dev/ttyACM0
id: leader
use_degrees: true
```

### Robot (`config/robot_so101.yaml`)
```yaml
type: so101_follower
port: /dev/ttyACM1
id: follower
use_degrees: true
disable_torque_on_disconnect: true
```

## LeRobot Commands

### Calibration
```bash
# Calibrate follower
python3 -m lerobot.calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower

# Calibrate leader
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

## Visualization
```bash
rqt_graph
ros2 run rqt_tf_tree rqt_tf_tree --force-discover
rviz2
```

# 
```
~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/leader.json
```
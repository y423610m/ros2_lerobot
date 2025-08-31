
# host setup(only once, depends on your environment)
```
xhost +
```

# build and run docker
```
docker compose up -d --build
docker exec -it ros2_lerobot bash
```

# build ros pkg
```
colcon build --symlink-install
source install/setup.bash
```

# run
```
ros2 launch my_robot_description view_bot.launch.py
ros2 launch my_robot_description view_bot_gui.launch.py
ros2 launch my_robot_description view_bot_py.launch.py
ros2 launch my_robot_description view_bot_gui_namespace.launch.py
```

# ros viz
```
rqt_graph
ros2 run rqt_tf_tree rqt_tf_tree --force-discover
```

# calib
```
python3 -m lerobot.calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower
# -> save to /root/.cache/huggingface/lerobot/calibration/robots/so101_follower/follower.json
```

# calib
```
python3 -m lerobot.calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader
# -> /root/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/leader.json  
```

# teleoperate
```
python3 -m lerobot.teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader
```
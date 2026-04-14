#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import argparse
import yaml
import draccus

from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.robots import RobotConfig


class LeRobotRobotNode(Node):
    def __init__(self, cfg: RobotConfig):
        super().__init__('lerobot_robot')

        self.robot = make_robot_from_config(cfg)
        self.robot.connect()

        self.subscription = self.create_subscription(
            JointState,
            'joint_commands',
            self.command_callback,
            10
        )

    def command_callback(self, msg: JointState):
        action = {}
        for joint_name, position in zip(msg.name, msg.position):
            action[f'{joint_name}.pos'] = position * 180.0 / 3.1415
        self.robot.send_action(action)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config YAML')
    known_args, _ = parser.parse_known_args()

    if known_args.config:
        with open(known_args.config, 'r') as f:
            yaml_dict = yaml.safe_load(f)
            cfg = draccus.decode(RobotConfig, yaml_dict)

    rclpy.init(args=args)
    node = LeRobotRobotNode(cfg)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

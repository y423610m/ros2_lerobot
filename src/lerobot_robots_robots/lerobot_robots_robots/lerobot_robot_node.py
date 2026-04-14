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
            'joint_command',
            self.command_callback,
            10
        )
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.publish_joint_states)

    def publish_joint_states(self):
        observation = self.robot.get_observation()

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = []
        msg.position = []
        msg.velocity = []
        msg.effort = []

        for key, value in observation.items():
            if key.endswith('.pos'):
                msg.name.append(key.replace('.pos', ''))
                msg.position.append(value * 3.1415 / 180.0)
                msg.velocity.append(0.0)
                msg.effort.append(0.0)

        self.publisher_.publish(msg)

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

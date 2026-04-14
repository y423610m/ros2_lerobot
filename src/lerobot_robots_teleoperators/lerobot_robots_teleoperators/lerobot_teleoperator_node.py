#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import argparse
import yaml
import draccus

from lerobot.teleoperators import make_teleoperator_from_config, TeleoperatorConfig
from lerobot.teleoperators import so_leader  # noqa: F401 - registers SO100/SO101


class LeRobotTeleoperatorNode(Node):
    def __init__(self, cfg: TeleoperatorConfig):
        super().__init__('lerobot_teleoperator')

        self.teleop = make_teleoperator_from_config(cfg)
        self.teleop.connect()

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.publish_joint_states)
        self.publisher_ = self.create_publisher(JointState, 'joint_commands', 10)

    def publish_joint_states(self):
        action = self.teleop.get_action()

        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.name = []
        msg.position = []
        msg.velocity = []
        msg.effort = []

        for joint_name, joint_value in action.items():
            msg.name.append(joint_name.replace('.pos', ''))
            msg.velocity.append(0.0)
            msg.effort.append(0.0)
            msg.position.append(joint_value * 3.1415 / 180.0)

        self.publisher_.publish(msg)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config YAML')
    known_args, _ = parser.parse_known_args()

    if known_args.config:
        with open(known_args.config, 'r') as f:
            yaml_dict = yaml.safe_load(f)
        cfg = draccus.decode(TeleoperatorConfig, yaml_dict)
    else:
        cfg = TeleoperatorConfig(type='so101_leader', port='/dev/ttyACM0')

    rclpy.init(args=args)
    node = LeRobotTeleoperatorNode(cfg)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

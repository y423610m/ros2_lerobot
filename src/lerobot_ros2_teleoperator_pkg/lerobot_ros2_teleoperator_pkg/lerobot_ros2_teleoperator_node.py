#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import sys
import yaml

from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.so_leader import SO101LeaderConfig


class SO101LeaderNode(Node):
    def __init__(self, cfg: SO101LeaderConfig):
        super().__init__('so101_leader')

        self.teleop = make_teleoperator_from_config(cfg)
        self.teleop.connect()

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.publish_joint_states)
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)

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
    config_path = None
    cli_args = sys.argv[1:]

    i = 0
    while i < len(cli_args):
        if cli_args[i] == '--config' and i + 1 < len(cli_args):
            config_path = cli_args[i + 1]
            break
        i += 1

    if config_path:
        with open(config_path, 'r') as f:
            yaml_dict = yaml.safe_load(f)
        cfg = SO101LeaderConfig(**yaml_dict)
    else:
        cfg = SO101LeaderConfig(port='/dev/ttyACM0')

    rclpy.init(args=args)
    node = SO101LeaderNode(cfg)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

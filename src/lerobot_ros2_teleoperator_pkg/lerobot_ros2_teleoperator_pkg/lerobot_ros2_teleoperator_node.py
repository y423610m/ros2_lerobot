#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math
import time
import random
import draccus

import json
import os

from urdf_parser_py.urdf import URDF

from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    # bi_so100_leader,
    # gamepad,
    # homunculus,
    # koch_leader,
    make_teleoperator_from_config,
    # so100_leader,
    so_leader,
)
from pathlib import Path


class SO101LeaderNode(Node):
    def __init__(self):
        super().__init__('so101_leader')

        # self.declare_parameter("robot_description")
        # xml_string = self.get_parameter('robot_description').get_parameter_value().string_value
        # self._model = URDF.from_xml_string(xml_string)
        # import IPython; IPython.embed()
        # self.declare_parameter('port')
        # self.declare_parameter('id')
        # self.declare_parameter('calibration_dir')
        # self.declare_parameter('use_degrees')
        # print(self.get_parameter('port'))
        cfg_teleop = so_leader.SO101LeaderConfig(
            port = '/dev/ttyACM1',
            id = 'id',
            # calibration_dir = Path(self.get_parameter('calibration_dir').value),
            # use_degrees = self.get_parameter('use_degrees').value
        )
        self.teleop = make_teleoperator_from_config(cfg_teleop)
        self.teleop.connect()

        # calibfile = '/'.join([self.get_parameter('calibration_dir').value, f"{self.get_parameter('id').value}.json"])
        # with open(calibfile, 'r') as f:
        #     self.calib = json.load(f)

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.publish_joint_states)

        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)

    def publish_joint_states(self):
        action = self.teleop.get_action()

        # self.get_logger().info(f'AAA: {self.calib=}')
        # self.get_logger().info(f"AAA: {action=} {[joint.name for joint in self._model.joints]}")

        # action["shoulder_pan.pos"] += 0
        # action["shoulder_lift.pos"] -= 90 + 10
        # action["elbow_flex.pos"] += 45 + 25
        # action["wrist_flex.pos"] += 60
        # action["wrist_roll.pos"] += 0
        # for joint_name, calib_info in self.calib.items():
        #     if joint_name == 'gripper':
        #         continue
        #     # action[joint_name+'.pos'] += calib_info.get('homing_offset', 0) * 360 / (calib_info.get('range_max', 4000)-calib_info.get('range_min', 0))
        #     action[joint_name+'.pos'] += calib_info.get('homing_offset', 0) * 200 / 4800

        # # self.get_logger().info(f'AAA: {action=}')
        # # self.get_logger().info(f"AAA: {action['shoulder_pan.pos']=}")
        
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()  # current time
        
        msg.name = []
        msg.velocity = []
        msg.effort = []


        # for joint in self._model.joints:
            # if joint.type not in ['revolute', 'prismatic']:
                # continue

        for joint_name, joint_value in action.items():
            msg.name.append(joint_name.replace('.pos', ''))
            msg.velocity.append(0)
            msg.effort.append(0)
            msg.position.append(joint_value * 3.1415 / 180.0)
                
        self.publisher_.publish(msg)
        # self.get_logger().info(f'Published joint positions: {msg.position}')

def main(args=None):
    rclpy.init(args=args)
    node = SO101LeaderNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


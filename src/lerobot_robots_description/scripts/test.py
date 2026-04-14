#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math
import time
import random

from urdf_parser_py.urdf import URDF
# robot = URDF.from_xml_file("urdf/SO101/so101_new_calib.urdf")
# robot = URDF.from_xml_string(xml_string)

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('test')

        self.declare_parameter("robot_description")
        xml_string = self.get_parameter('robot_description').get_parameter_value().string_value
        self._model = URDF.from_xml_string(xml_string)

        # Publisher to the "joint_states" topic
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        
        # Timer to call publish function periodically (10 Hz)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.publish_joint_states)
        
        self.joint_angle = 0.0  # initial joint angle

    def publish_joint_states(self):
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()  # current time
        
        msg.name = []
        msg.velocity = []
        msg.effort = []
        for joint in self._model.joints:
            if joint.type not in ['revolute', 'prismatic']:
                continue
            msg.name.append(joint.name)
            msg.velocity.append(0)
            msg.effort.append(0)
            msg.position.append(random.random())
                
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published joint positions: {msg.position}')

def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

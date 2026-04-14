#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class LeaderFollowerNode(Node):
    def __init__(self):
        super().__init__('leader_follower')

        # Subscribes to leader joint states
        self.subscription = self.create_subscription(
            JointState,
            '/leader/joint_states',
            self.leader_callback,
            10
        )

        # Publishes to follower joint commands
        self.publisher = self.create_publisher(
            JointState,
            '/follower/joint_command',
            10
        )
        
        self.get_logger().info('Leader-Follower bridge node started.')
        self.get_logger().info('Subscribed to: /leader/joint_states')
        self.get_logger().info('Publishing to: /follower/joint_command')

    def leader_callback(self, msg: JointState):
        # Simply relay the joint state as a command
        # You might want to add some processing here if needed
        command_msg = JointState()
        command_msg.header.stamp = self.get_clock().now().to_msg()
        command_msg.name = msg.name
        command_msg.position = msg.position
        command_msg.velocity = msg.velocity
        command_msg.effort = msg.effort
        
        self.publisher.publish(command_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LeaderFollowerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

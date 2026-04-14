#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import tf2_ros
import os

class RobotTFPublisher(Node):
    def __init__(self):
        super().__init__('robot_tf_publisher')

        # Get robot_id from parameter
        self.declare_parameter('robot_id', 'robot1')
        robot_id = self.get_parameter('robot_id').get_parameter_value().string_value

        # Example positions for each robot (could also be passed as parameters)
        if robot_id == 'robot1':
            base_pos = (0.0, 0.0, 0.0)
        elif robot_id == 'robot2':
            base_pos = (1.0, 0.0, 0.0)
        else:
            base_pos = (0.0, 0.0, 1.0)

        # Create static transform broadcaster
        self.broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        transforms = []

        # base_link
        t_base = TransformStamped()
        t_base.header.stamp = self.get_clock().now().to_msg()
        t_base.header.frame_id = 'world'
        t_base.child_frame_id = f'{robot_id}/base'
        # t_base.child_frame_id = f'{robot_id}/base_link'
        t_base.transform.translation.x = base_pos[0]
        t_base.transform.translation.y = base_pos[1]
        t_base.transform.translation.z = base_pos[2]
        t_base.transform.rotation.w = 1.0
        transforms.append(t_base)

        # camera_link
        # t_camera = TransformStamped()
        # t_camera.header.stamp = self.get_clock().now().to_msg()
        # t_camera.header.frame_id = f'{robot_id}/base_link'
        # t_camera.child_frame_id = f'{robot_id}/camera_link'
        # t_camera.transform.translation.x = 0.2
        # t_camera.transform.translation.y = 0.0
        # t_camera.transform.translation.z = 0.5
        # t_camera.transform.rotation.w = 1.0
        # transforms.append(t_camera)

        self.broadcaster.sendTransform(transforms)
        self.get_logger().info(f"Static TF published for {robot_id} {base_pos=}")

def main(args=None):
    rclpy.init(args=args)
    node = RobotTFPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

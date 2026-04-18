#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class SO101JointStateToTrajectory(Node):
    def __init__(self):
        super().__init__('so101_joint_state_to_trajectory')
        self.target_joints = [
            'shoulder_pan', 'shoulder_lift', 'elbow_flex',
            'wrist_flex', 'wrist_roll', 'gripper'
        ]

        self.sub = self.create_subscription(
            JointState, '/follower/joint_command', self.callback, 10
        )
        self.pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10
        )
        self.get_logger().info('SO101 JointState → JointTrajectory bridge active')

    def callback(self, msg: JointState):
        traj = JointTrajectory()
        traj.joint_names = self.target_joints

        point = JointTrajectoryPoint()

        if msg.name and msg.position:
            pos_map = dict(zip(msg.name, msg.position))
            point.positions = [pos_map.get(j, 0.0) for j in self.target_joints]
        elif len(msg.position) == len(self.target_joints):
            point.positions = list(msg.position)
        else:
            self.get_logger().warn(f'Position size mismatch: {len(msg.position)} vs {len(self.target_joints)}')
            return

        point.time_from_start = Duration(sec=0, nanosec=5_000_000)
        traj.points = [point]

        self.pub.publish(traj)


def main(args=None):
    rclpy.init(args=args)
    node = SO101JointStateToTrajectory()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
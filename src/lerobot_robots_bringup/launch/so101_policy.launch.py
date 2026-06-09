"""Bring up the SO-101 arm node, two USB-camera publishers, and the trained
mjlab vision policy inference node together.

Usage:
    ros2 launch lerobot_robots_bringup so101_policy.launch.py \\
        checkpoint:=/abs/path/to/policy.jit

Override camera devices / topics as needed:
    ... wrist_device:=0 top_device:=1 control_hz:=50.0
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    arm_config = LaunchConfiguration('arm_config').perform(context)
    if not arm_config:
        arm_config = os.path.join(
            get_package_share_directory('lerobot_robots_robots'),
            'config',
            'robot_so101.yaml',
        )

    zero_action = LaunchConfiguration('zero_action').perform(context).lower() in ('1', 'true', 'yes')

    checkpoint = LaunchConfiguration('checkpoint').perform(context)
    if not checkpoint and not zero_action:
        raise RuntimeError(
            "checkpoint:= argument is required (path to a TorchScript .jit "
            "exported from mjlab via scripts/export_to_jit.py)"
        )

    wrist_device = int(LaunchConfiguration('wrist_device').perform(context))
    top_device = int(LaunchConfiguration('top_device').perform(context))
    cam_width = int(LaunchConfiguration('cam_width').perform(context))
    cam_height = int(LaunchConfiguration('cam_height').perform(context))
    cam_fps = float(LaunchConfiguration('cam_fps').perform(context))
    control_hz = float(LaunchConfiguration('control_hz').perform(context))
    device = LaunchConfiguration('device').perform(context)

    arm = Node(
        package='lerobot_robots_robots',
        executable='lerobot_robot_node',
        name='lerobot_robot',
        namespace='follower',
        output='screen',
        arguments=['--config', arm_config],
    )

    wrist_cam = Node(
        package='lerobot_robots_inference',
        executable='camera_publisher_node',
        name='wrist_camera',
        output='screen',
        parameters=[{
            'device_index': wrist_device,
            'topic_name': '/wrist_cam/image_raw',
            'frame_id': 'wrist_cam',
            'width': cam_width,
            'height': cam_height,
            'fps': cam_fps,
        }],
    )

    top_cam = Node(
        package='lerobot_robots_inference',
        executable='camera_publisher_node',
        name='top_camera',
        output='screen',
        parameters=[{
            'device_index': top_device,
            'topic_name': '/top_cam/image_raw',
            'frame_id': 'top_cam',
            'width': cam_width,
            'height': cam_height,
            'fps': cam_fps,
        }],
    )

    policy = Node(
        package='lerobot_robots_inference',
        executable='policy_node',
        name='policy',
        output='screen',
        parameters=[{
            'checkpoint_path': checkpoint,
            'control_hz': control_hz,
            'device': device,
            'zero_action': zero_action,
            'joint_states_topic': '/follower/joint_states',
            'joint_command_topic': '/follower/joint_command',
            'wrist_cam_topic': '/wrist_cam/image_raw',
            'top_cam_topic': '/top_cam/image_raw',
        }],
    )

    return [arm, wrist_cam, top_cam, policy]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('arm_config', default_value=''),
        DeclareLaunchArgument('checkpoint', default_value='',
                              description='Path to TorchScript .jit policy'),
        DeclareLaunchArgument('wrist_device', default_value='0'),
        DeclareLaunchArgument('top_device', default_value='1'),
        DeclareLaunchArgument('cam_width', default_value='640'),
        DeclareLaunchArgument('cam_height', default_value='480'),
        DeclareLaunchArgument('cam_fps', default_value='30.0'),
        DeclareLaunchArgument('control_hz', default_value='50.0',
                              description='Policy inference rate (Hz)'),
        DeclareLaunchArgument('device', default_value='auto',
                              description='Torch device: auto | cpu | cuda'),
        DeclareLaunchArgument('zero_action', default_value='false',
                              description='Diagnostic: ignore policy, command a zero action (= home pose)'),
        OpaqueFunction(function=launch_setup),
    ])

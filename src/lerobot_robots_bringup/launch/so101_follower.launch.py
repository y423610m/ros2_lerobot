import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    config_path = LaunchConfiguration('config').perform(context)

    if not config_path:
        config_path = os.path.join(
            get_package_share_directory('lerobot_robots_bringup'),
            'config',
            'robot_so101.yaml'
        )

    robot_node = Node(
        package='lerobot_robots_robots',
        executable='lerobot_robot_node',
        name='lerobot_robot',
        namespace='follower',
        output='screen',
        arguments=['--config', config_path],
    )

    return [robot_node]


def generate_launch_description():
    config_arg = DeclareLaunchArgument(
        'config',
        default_value='',
        description='Path to robot config YAML'
    )

    return LaunchDescription([
        config_arg,
        OpaqueFunction(function=launch_setup),
    ])

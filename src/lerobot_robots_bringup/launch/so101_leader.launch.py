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
            'teleop_so101.yaml'
        )

    teleoperator_node = Node(
        package='lerobot_robots_teleoperators',
        executable='lerobot_teleoperator_node',
        name='lerobot_teleoperator',
        namespace='leader',
        output='screen',
        arguments=['--config', config_path],
    )

    return [teleoperator_node]


def generate_launch_description():
    config_arg = DeclareLaunchArgument(
        'config',
        default_value='',
        description='Path to teleoperator config YAML'
    )

    return LaunchDescription([
        config_arg,
        OpaqueFunction(function=launch_setup),
    ])

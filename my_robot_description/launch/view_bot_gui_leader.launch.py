from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

import draccus
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    gamepad,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)

def generate_launch_description():
    urdf_path = os.path.join(
        get_package_share_directory('my_robot_description'),
        'urdf',
        'SO101',
        'so101_new_calib.urdf'
    )
    with open(urdf_path, 'r') as infp:
        robot_description_content = infp.read()

    pkg_share = get_package_share_directory('my_robot_description')
    config_path_leader = os.path.join(pkg_share, 'config', 'so101_leader.yaml')

    print(f"CCC {config_path_leader=}")

    return LaunchDescription([
        Node(
            package='my_robot_description',
            executable='so101_leader',
            name='so101_leader',
            parameters=[config_path_leader, {'robot_description': robot_description_content}]
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description_content}],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen'
        )
    ])


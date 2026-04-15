import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    # Get the URDF file from lerobot_robots_description
    pkg_description_share = get_package_share_directory('lerobot_robots_description')
    urdf_file = os.path.join(pkg_description_share, 'urdf', 'SO101', 'so101_new_calib.urdf')
    
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # Robot State Publisher for the follower arm
    # It will subscribe to /follower/joint_states and publish the TF tree for the /follower namespace
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace='follower',
        output='screen',
        parameters=[{'robot_description': robot_description}],
    )

    # Launch RViz 2
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(pkg_description_share, 'rviz', 'view_follower.rviz')]
    )

    return [
        robot_state_publisher_node,
        rviz_node
    ]


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup),
    ])

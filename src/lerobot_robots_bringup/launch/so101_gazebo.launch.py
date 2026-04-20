import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable

def generate_launch_description():
    pkg_description = get_package_share_directory('lerobot_robots_description')
    
    # Set gazebo resource path
    install_dir = get_package_share_directory('lerobot_robots_description')
    parent_dir = os.path.dirname(install_dir)
    
    gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=[os.path.join(parent_dir)]
    )
    
    urdf_file = os.path.join(pkg_description, 'urdf', 'SO101', 'so101_gazebo.urdf')
    with open(urdf_file, 'r') as infp:
        robot_description_config = infp.read()
    robot_description_config = robot_description_config.replace(
        '$(find lerobot_robots_description)/config/so101_controllers.yaml',
        f"{install_dir}/config/so101_controllers.yaml"
    )

    world_file = os.path.join(pkg_description, 'worlds', 'lerobot.sdf')

    # Gazebo Sim
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
        launch_arguments={'gz_args': f'-r {world_file}'}.items(),
    )

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[{'robot_description': robot_description_config}],
    )

    # Spawn Robot
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'so101',
            '-topic', 'robot_description',
            '-x', '-0.4',
            '-y', '0.25',
            '-z', '0.8',
            '--ros-args', '--log-level', 'debug'
        ],
        output='screen',
    )

    # Bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/camera@sensor_msgs/msg/Image[gz.msgs.Image',
        ],
        output='screen',
    )

    # Controllers
    load_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'joint_state_broadcaster'],
        output='screen'
    )

    load_joint_trajectory_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'joint_trajectory_controller'],
        output='screen'
    )

    so101_joint_state_to_trajectory = Node(
        package='lerobot_robots_bringup',
        executable='so101_joint_state_to_trajectory.py',
        output='screen',
    )

    return LaunchDescription([
        gz_resource_path,
        gazebo,
        robot_state_publisher,
        spawn_robot,
        bridge,
        so101_joint_state_to_trajectory,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_robot,
                on_exit=[load_joint_state_broadcaster],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_joint_state_broadcaster,
                on_exit=[load_joint_trajectory_controller],
            )
        ),
    ])

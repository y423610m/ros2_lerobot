from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    urdf_path = os.path.join(
        get_package_share_directory('my_robot_description'),
        'urdf',
        'SO101',
        'so101_new_calib.urdf'
    )
    with open(urdf_path, 'r') as infp:
        robot_description_content = infp.read()

    namespace1 = 'robot1'
    namespace2 = 'robot2'

    return LaunchDescription([
        # robot1
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name=f'{namespace1}_joint_state_publisher_gui',
            namespace=namespace1,
            parameters=[{'robot_description': robot_description_content}],
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name=f'{namespace1}_robot_state_publisher',
            namespace=namespace1,
            parameters=[{
                'robot_description': robot_description_content,
                'frame_prefix': f'{namespace1}/'
            }],
        ),
        Node(
            package='my_robot_description',
            executable='robot_tf_node',
            name=f'{namespace1}_tf',
            parameters=[{'robot_id': namespace1}]
        ),

        # robot2
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name=f'{namespace2}_joint_state_publisher_gui',
            namespace=namespace2,
            parameters=[{'robot_description': robot_description_content}],
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name=f'{namespace2}_robot_state_publisher',
            namespace=namespace2,
            parameters=[{
                'robot_description': robot_description_content,
                'frame_prefix': f'{namespace2}/'
            }],
        ),
        Node(
            package='my_robot_description',
            executable='robot_tf_node',
            name=f'{namespace2}_tf',
            parameters=[{'robot_id': namespace2}]
        ),

        # viz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen'
        )
    ])


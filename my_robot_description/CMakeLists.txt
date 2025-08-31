cmake_minimum_required(VERSION 3.5)
project(my_robot_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch urdf
  DESTINATION share/my_robot_description/
)

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(urdf REQUIRED)

add_executable(random_joint_publisher src/random_joint_publisher.cpp)
ament_target_dependencies(random_joint_publisher rclcpp sensor_msgs urdf)
install(TARGETS random_joint_publisher DESTINATION lib/my_robot_description)

# --- Add Python support ---
find_package(ament_cmake_python REQUIRED)

# Install Python modules
ament_python_install_package(my_robot_description)

# Install Python scripts
install(PROGRAMS
  scripts/test.py
  DESTINATION lib/my_robot_description
  RENAME test
)

install(PROGRAMS
  scripts/robot_tf_node.py
  DESTINATION lib/my_robot_description
  RENAME robot_tf_node
)

ament_package()


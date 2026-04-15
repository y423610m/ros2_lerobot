cmake_minimum_required(VERSION 3.8)
project(lerobot_robots_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

# Install URDF files
install(DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL)

# Install config files
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL)

# Install calibration files
install(DIRECTORY calib
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL)

# Install rviz
install(DIRECTORY calib
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL)

# Install scripts
install(PROGRAMS
  scripts/camera_test.py
  scripts/load_urdf.py
  scripts/robot_tf_node.py
  scripts/so101_leader.py
  scripts/test.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install scene.xml
install(FILES scene.xml
  DESTINATION share/${PROJECT_NAME}
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

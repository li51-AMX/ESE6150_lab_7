cmake_minimum_required(VERSION 3.5)
project(real_sense)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependancies
foreach(PACKAGE 
  ament_cmake 
  ament_cmake_python
  rclcpp
  rclpy 
  cv_bridge
  sensor_msgs
  OpenCV
  realsense2
  visualization_msgs

  # Might need later
  std_msgs
  geometry_msgs
  Eigen3
  tf2_ros)

  find_package(${PACKAGE} REQUIRED)
endforeach()

# ========= CPP =========
# Include Cpp "include" directory
include_directories(include)
include_directories(
  ${EIGEN3_INCLUDE_DIRS}
)

# Create Cpp executable
add_executable(camera_node src/camera_node.cpp)

ament_target_dependencies(camera_node 
  rclcpp sensor_msgs cv_bridge
)

target_link_libraries(camera_node
  ${OpenCV_LIBS}
  realsense2
)

# Install Cpp executables
install(TARGETS
  camera_node
  DESTINATION lib/${PROJECT_NAME})

# ========= PYTHON =========
# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/camera_node.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
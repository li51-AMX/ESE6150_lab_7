cmake_minimum_required(VERSION 3.5)
project(vision_integration_pkg)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

install(PROGRAMS
  scripts/integrated.py
  scripts/lane.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  model/model_fp16.trt
  DESTINATION lib/${PROJECT_NAME}/model
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

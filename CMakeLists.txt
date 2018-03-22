cmake_minimum_required(VERSION 2.8.3)
project(edge_grasp)

find_package(catkin REQUIRED COMPONENTS geometry_msgs)

catkin_python_setup()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

cmake_minimum_required(VERSION 3.0.2)
project(keyboard_control)

add_compile_options(-std=c++11)
add_compile_options(-o1)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  message_generation
)

generate_messages(
	DEPENDENCIES
	geometry_msgs
)

find_package(roslaunch)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(KeyBoardControl
  src/KeyBoardControl.cpp
)
target_link_libraries(KeyBoardControl
  ${catkin_LIBRARIES} 
)

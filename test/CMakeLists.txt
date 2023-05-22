cmake_minimum_required(VERSION 3.0.2)
project(test)

add_compile_options(-std=c++11)
add_compile_options(-o2)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs 
	geometry_msgs 
	message_generation
  sensor_msgs
  cv_bridge
  image_transport
)

generate_messages(
	DEPENDENCIES
	geometry_msgs
)

catkin_package(CATKIN_DEPENDS message_runtime)

find_package(roslaunch)

include_directories(
  src
  src/FSM
  src/FSM/TaskController
  ${catkin_INCLUDE_DIRS}
)

file(GLOB FSM_TASK src/FSM/TaskController/**.cpp)
file(GLOB FSM_TASKCFSM src/FSM/TaskController/TaskCFSM/**.cpp)

add_executable(KeyBoardControlNode 
  src/KeyBoardControl.cpp
)
target_link_libraries(KeyBoardControlNode 
  ${catkin_LIBRARIES} 
)

add_executable(CVNode 
  src/opencv.cpp
)
target_link_libraries(CVNode 
  ${catkin_LIBRARIES} 
)

add_executable(TaskNode 
  src/main.cpp
  ${FSM_TASK}
  ${FSM_TASKCFSM}
)
target_link_libraries(TaskNode 
  ${catkin_LIBRARIES} 
)


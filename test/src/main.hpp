#ifndef MAIN_HPP
#define MAIN_HPP

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose2D.h>
#include <std_msgs/Bool.h>

class TaskController;

enum class T3Area : uint8_t
{
  AreaA = 0,
  AreaB = 1,
  AreaC = 2,
  AreaD = 3
};

class MainNode
{
public:
  ros::NodeHandle *pNodeHandle;
  TaskController *task_controller;

  std_msgs::Bool laser_switch;
  std_msgs::Bool camera_switch;

  ros::Publisher laser_switch_pub;
  ros::Publisher camera_switch_pub;

  int task;

  T3Area area;
  T3Area last_area;

  double pos_x;
  double pos_y;
  double pos_theta;

  MainNode(/* args */) : task(0), pos_x(0.0), pos_y(0.0), pos_theta(0.0), area(T3Area::AreaA), last_area(T3Area::AreaA)
  {
  }

  void Init(ros::NodeHandle *_pNodeHandle) { pNodeHandle = _pNodeHandle; }

  void UpdateSwitch();
  void UpdateParam();
  void PubArea();

  ~MainNode()
  {
  }
};

#endif
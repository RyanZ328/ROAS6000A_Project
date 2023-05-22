#ifndef TASK_A_STATE_HPP
#define TASK_A_STATE_HPP

#include <ros/ros.h>
#include "TaskController.hpp"
#include <geometry_msgs/Twist.h>

class SmartCarKeyboardTeleopNode
{
public:
    double walk_vel_;
    double run_vel_;
    double yaw_rate_;
    double yaw_rate_run_;

    ros::NodeHandle *pn;
    ros::Publisher pub_;
    geometry_msgs::Twist cmdvel_;

    SmartCarKeyboardTeleopNode()
    {
            
    }

    ~SmartCarKeyboardTeleopNode() {}

    void Init(ros::NodeHandle *pn);

    void keyboardLoop();
    void stopRobot()
    {
        cmdvel_.linear.x = 0.0;
        cmdvel_.angular.z = 0.0;
        pub_.publish(cmdvel_);
    }
};

class TaskAState : public State<TaskController>
{
private:
public:
    bool pub_flag = false;

    virtual void Init(TaskController *pOwner);
    virtual void Enter(TaskController *pOwner);
    virtual void Execute(TaskController *pOwner);
    virtual void Exit(TaskController *pOwner);

    static TaskAState *Instance()
    {
        static TaskAState instance;
        return &instance;
    }
};

#endif

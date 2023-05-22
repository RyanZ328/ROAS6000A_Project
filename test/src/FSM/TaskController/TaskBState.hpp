#ifndef TASK_B_STATE_HPP
#define TASK_B_STATE_HPP

#include <ros/ros.h>
#include "TaskController.hpp"

class TaskBState : public State<TaskController>
{
private:
public:
    virtual void Init(TaskController *pOwner);
    virtual void Enter(TaskController *pOwner);
    virtual void Execute(TaskController *pOwner);
    virtual void Exit(TaskController *pOwner);

    static TaskBState *Instance()
    {
        static TaskBState instance;
        return &instance;
    }
};

#endif

#ifndef TASK_D_STATE_HPP
#define TASK_D_STATE_HPP

#include <ros/ros.h>
#include "TaskController.hpp"

class TaskDState : public State<TaskController>
{
private:
public:
    virtual void Init(TaskController *pOwner);
    virtual void Enter(TaskController *pOwner);
    virtual void Execute(TaskController *pOwner);
    virtual void Exit(TaskController *pOwner);

    static TaskDState *Instance()
    {
        static TaskDState instance;
        return &instance;
    }
};

#endif

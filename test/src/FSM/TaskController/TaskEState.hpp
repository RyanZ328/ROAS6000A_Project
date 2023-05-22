#ifndef TASK_E_STATE_HPP
#define TASK_E_STATE_HPP

#include <ros/ros.h>
#include "TaskController.hpp"

class TaskEState : public State<TaskController>
{
private:
public:
    virtual void Init(TaskController *pOwner);
    virtual void Enter(TaskController *pOwner);
    virtual void Execute(TaskController *pOwner);
    virtual void Exit(TaskController *pOwner);

    static TaskEState *Instance()
    {
        static TaskEState instance;
        return &instance;
    }
};

#endif

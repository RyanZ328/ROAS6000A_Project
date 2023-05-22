#ifndef TASK_C_AREA_B_STATE_HPP
#define TASK_C_AREA_B_STATE_HPP

#include <ros/ros.h>
#include "../TaskController.hpp"

class TaskCAreaBState : public State<TaskController>
{
private:
    

public:
    virtual void Init(TaskController *pOwner);
    virtual void Enter(TaskController *pOwner);
    virtual void Execute(TaskController *pOwner);
    virtual void Exit(TaskController *pOwner);

    static TaskCAreaBState *Instance()
    {
        static TaskCAreaBState instance;
        return &instance;
    }
};

#endif

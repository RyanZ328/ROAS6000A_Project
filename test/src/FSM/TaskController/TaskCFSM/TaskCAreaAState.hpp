#ifndef TASK_C_AREA_A_STATE_HPP
#define TASK_C_AREA_A_STATE_HPP

#include <ros/ros.h>
#include "../TaskController.hpp"

class TaskCAreaAState : public State<TaskController>
{
private:
    

public:
    virtual void Init(TaskController *pOwner);
    virtual void Enter(TaskController *pOwner);
    virtual void Execute(TaskController *pOwner);
    virtual void Exit(TaskController *pOwner);

    static TaskCAreaAState *Instance()
    {
        static TaskCAreaAState instance;
        return &instance;
    }
};

#endif

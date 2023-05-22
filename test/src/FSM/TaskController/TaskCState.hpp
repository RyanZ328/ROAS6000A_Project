#ifndef TASK_C_STATE_HPP
#define TASK_C_STATE_HPP

#include <ros/ros.h>
#include "TaskController.hpp"

class TaskCFsm : public StateMachine<TaskController>
{
public:
    TaskCFsm() : StateMachine<TaskController>(){}

    void Init(TaskController *pOwner);
    void Init();
};

class TaskCState : public State<TaskController>
{
private:
    

public:
    virtual void Init(TaskController *pOwner);
    virtual void Enter(TaskController *pOwner);
    virtual void Execute(TaskController *pOwner);
    virtual void Exit(TaskController *pOwner);

    TaskCFsm task_c_fsm;

    static TaskCState *Instance()
    {
        static TaskCState instance;
        return &instance;
    }
};

#endif

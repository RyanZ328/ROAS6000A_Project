#ifndef TASK_NULL_STATE_HPP
#define TASK_NULL_STATE_HPP

#include "TaskController.hpp"

class TaskNULLState : public State<TaskController>
{
private:
public:
    virtual void Init(TaskController *pOwner);
    virtual void Enter(TaskController *pOwner);
    virtual void Execute(TaskController *pOwner);
    virtual void Exit(TaskController *pOwner);

    static TaskNULLState *Instance()
    {
        static TaskNULLState instance;
        return &instance;
    }
};

#endif

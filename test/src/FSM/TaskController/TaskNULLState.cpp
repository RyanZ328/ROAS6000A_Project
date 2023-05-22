#include "TaskNULLState.hpp"

void TaskNULLState::Init(TaskController *pOwner)
{
    puts("TaskNULLStateInit");
}

void TaskNULLState::Enter(TaskController *pOwner)
{
    puts("TaskNULLStateEnter");
}

void TaskNULLState::Execute(TaskController *pOwner)
{
}

void TaskNULLState::Exit(TaskController *pOwner)
{
    puts("TaskNULLStateExit");
}

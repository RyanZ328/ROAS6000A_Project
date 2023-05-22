#include "TaskBState.hpp"
#include "TaskAState.hpp"

void TaskBState::Init(TaskController *pOwner)
{

}

void TaskBState::Enter(TaskController *pOwner)
{
    TaskAState::Instance()->pub_flag = true;

    pOwner->p_main_node->camera_switch.data = 0;
    pOwner->p_main_node->laser_switch.data = 1;
}

void TaskBState::Execute(TaskController *pOwner)
{

}

void TaskBState::Exit(TaskController *pOwner)
{
    TaskAState::Instance()->pub_flag = false;

    pOwner->p_main_node->camera_switch.data = 0;
    pOwner->p_main_node->laser_switch.data = 0;
}

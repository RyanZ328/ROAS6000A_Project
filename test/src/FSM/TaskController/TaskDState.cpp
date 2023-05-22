#include "TaskDState.hpp"
#include "TaskAState.hpp"

void TaskDState::Init(TaskController *pOwner)
{

}

void TaskDState::Enter(TaskController *pOwner)
{
    TaskAState::Instance()->pub_flag = true;

    pOwner->p_main_node->camera_switch.data = 1;
    pOwner->p_main_node->laser_switch.data = 1;
}

void TaskDState::Execute(TaskController *pOwner)
{

}

void TaskDState::Exit(TaskController *pOwner)
{
    TaskAState::Instance()->pub_flag = false;

    pOwner->p_main_node->camera_switch.data = 0;
    pOwner->p_main_node->laser_switch.data = 1;
}

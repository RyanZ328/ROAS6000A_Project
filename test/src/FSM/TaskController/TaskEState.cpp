#include "TaskEState.hpp"
#include "TaskAState.hpp"

void TaskEState::Init(TaskController *pOwner)
{

}

void TaskEState::Enter(TaskController *pOwner)
{
    TaskAState::Instance()->pub_flag = true;

    pOwner->p_main_node->camera_switch.data = 1;
    pOwner->p_main_node->laser_switch.data = 0;
}

void TaskEState::Execute(TaskController *pOwner)
{

}

void TaskEState::Exit(TaskController *pOwner)
{
    TaskAState::Instance()->pub_flag = false;

    pOwner->p_main_node->camera_switch.data = 0;
    pOwner->p_main_node->laser_switch.data = 0;
}

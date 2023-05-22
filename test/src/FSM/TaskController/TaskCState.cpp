#include "TaskCState.hpp"
#include "TaskAState.hpp"

void TaskCFsm::Init(TaskController *pOwner)
{
    m_pOwner = pOwner;
    Init();
}

void TaskCFsm::Init()
{
    
}

void TaskCState::Init(TaskController *pOwner)
{

}

void TaskCState::Enter(TaskController *pOwner)
{
    TaskAState::Instance()->pub_flag = true;

    pOwner->p_main_node->camera_switch.data = 0;
    pOwner->p_main_node->laser_switch.data = 1;
}

void TaskCState::Execute(TaskController *pOwner)
{
    task_c_fsm.Update();
}

void TaskCState::Exit(TaskController *pOwner)
{
    TaskAState::Instance()->pub_flag = false;

    pOwner->p_main_node->camera_switch.data = 0;
    pOwner->p_main_node->laser_switch.data = 0;
}

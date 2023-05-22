#include "TaskCAreaDState.hpp"
#include "TaskCAreaCDState.hpp"
#include "../TaskCState.hpp"

void TaskCAreaDState::Init(TaskController *pOwner)
{

}

void TaskCAreaDState::Enter(TaskController *pOwner)
{
    pOwner->p_main_node->area = T3Area::AreaD;
}

void TaskCAreaDState::Execute(TaskController *pOwner)
{
    Point pos(pOwner->p_main_node->pos_x,pOwner->p_main_node->pos_y);

    
    if(EdgeCheck(pos,Point5,Point6))
    {
        TaskCState::Instance()->task_c_fsm.ChangeState(TaskCAreaCDState::Instance());
        return;
    }
}

void TaskCAreaDState::Exit(TaskController *pOwner)
{
    pOwner->p_main_node->last_area = T3Area::AreaD;
}
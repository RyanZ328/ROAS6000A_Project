#include "TaskCAreaCState.hpp"
#include "TaskCAreaBCState.hpp"
#include "TaskCAreaCDState.hpp"
#include "../TaskCState.hpp"

void TaskCAreaCState::Init(TaskController *pOwner)
{

}

void TaskCAreaCState::Enter(TaskController *pOwner)
{
    pOwner->p_main_node->area = T3Area::AreaC;
}

void TaskCAreaCState::Execute(TaskController *pOwner)
{
    Point pos(pOwner->p_main_node->pos_x,pOwner->p_main_node->pos_y);

    if(EdgeCheck(pos,Point3,Point4))
    {
        TaskCState::Instance()->task_c_fsm.ChangeState(TaskCAreaBCState::Instance());
        return;
    }
    
    if(EdgeCheck(pos,Point5,Point6))
    {
        TaskCState::Instance()->task_c_fsm.ChangeState(TaskCAreaCDState::Instance());
        return;
    }
}

void TaskCAreaCState::Exit(TaskController *pOwner)
{
    pOwner->p_main_node->last_area = T3Area::AreaC;
}

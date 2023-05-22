#include "TaskCAreaBState.hpp"
#include "TaskCAreaBCState.hpp"
#include "TaskCAreaABState.hpp"
#include "../TaskCState.hpp"

void TaskCAreaBState::Init(TaskController *pOwner)
{
    
}

void TaskCAreaBState::Enter(TaskController *pOwner)
{
    pOwner->p_main_node->area = T3Area::AreaB;
}

void TaskCAreaBState::Execute(TaskController *pOwner)
{
    Point pos(pOwner->p_main_node->pos_x,pOwner->p_main_node->pos_y);

    if(EdgeCheck(pos,Point1,Point2))
    {
        TaskCState::Instance()->task_c_fsm.ChangeState(TaskCAreaABState::Instance());
        return;
    }
    
    if(EdgeCheck(pos,Point3,Point4))
    {
        TaskCState::Instance()->task_c_fsm.ChangeState(TaskCAreaBCState::Instance());
        return;
    }
}

void TaskCAreaBState::Exit(TaskController *pOwner)
{
    pOwner->p_main_node->last_area = T3Area::AreaB;
}

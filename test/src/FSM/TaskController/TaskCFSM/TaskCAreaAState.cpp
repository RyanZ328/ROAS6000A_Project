#include "TaskCAreaABState.hpp"
#include "TaskCAreaAState.hpp"
#include "../TaskCState.hpp"

void TaskCAreaAState::Init(TaskController *pOwner)
{

}

void TaskCAreaAState::Enter(TaskController *pOwner)
{
    pOwner->p_main_node->area = T3Area::AreaA;
}

void TaskCAreaAState::Execute(TaskController *pOwner)
{
    Point pos(pOwner->p_main_node->pos_x,pOwner->p_main_node->pos_y);

    if(EdgeCheck(pos,Point1,Point2))
    {
        TaskCState::Instance()->task_c_fsm.ChangeState(TaskCAreaABState::Instance());
        return;
    }
}

void TaskCAreaAState::Exit(TaskController *pOwner)
{
    pOwner->p_main_node->last_area = T3Area::AreaA;
}

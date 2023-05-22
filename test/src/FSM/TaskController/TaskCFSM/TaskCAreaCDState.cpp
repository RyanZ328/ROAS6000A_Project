#include "TaskCAreaCDState.hpp"
#include "TaskCAreaCState.hpp"
#include "TaskCAreaDState.hpp"
#include "../TaskCState.hpp"

void TaskCAreaCDState::Init(TaskController *pOwner)
{

}

void TaskCAreaCDState::Enter(TaskController *pOwner)
{
    pOwner->p_main_node->area = pOwner->p_main_node->last_area;
}

void TaskCAreaCDState::Execute(TaskController *pOwner)
{
    Point pos(pOwner->p_main_node->pos_x, pOwner->p_main_node->pos_y);

    if (!EdgeCheck(pos, Point5, Point6))
    {
        if (AreaCheck(pos, area_c))
        {
            TaskCState::Instance()->task_c_fsm.ChangeState(TaskCAreaCState::Instance());
        }
        else
        {
            TaskCState::Instance()->task_c_fsm.ChangeState(TaskCAreaDState::Instance());
        }
    }
}

void TaskCAreaCDState::Exit(TaskController *pOwner)
{
    
}

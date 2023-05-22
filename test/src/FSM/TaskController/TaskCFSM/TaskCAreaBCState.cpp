#include "TaskCAreaBState.hpp"
#include "TaskCAreaBCState.hpp"
#include "TaskCAreaCState.hpp"
#include "../TaskCState.hpp"

void TaskCAreaBCState::Init(TaskController *pOwner)
{
}

void TaskCAreaBCState::Enter(TaskController *pOwner)
{
    pOwner->p_main_node->area = pOwner->p_main_node->last_area;
}

void TaskCAreaBCState::Execute(TaskController *pOwner)
{
    Point pos(pOwner->p_main_node->pos_x, pOwner->p_main_node->pos_y);

    if (!EdgeCheck(pos, Point3, Point4))
    {
        if (AreaCheck(pos, area_b))
        {
            TaskCState::Instance()->task_c_fsm.ChangeState(TaskCAreaBState::Instance());
        }
        else
        {
            TaskCState::Instance()->task_c_fsm.ChangeState(TaskCAreaCState::Instance());
        }
    }
}

void TaskCAreaBCState::Exit(TaskController *pOwner)
{

}

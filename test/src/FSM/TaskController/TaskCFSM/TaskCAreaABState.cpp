#include "TaskCAreaABState.hpp"
#include "TaskCAreaAState.hpp"
#include "TaskCAreaBState.hpp"
#include "../TaskCState.hpp"

void TaskCAreaABState::Init(TaskController *pOwner)
{

}

void TaskCAreaABState::Enter(TaskController *pOwner)
{
    pOwner->p_main_node->area = pOwner->p_main_node->last_area; 
}

void TaskCAreaABState::Execute(TaskController *pOwner)
{
    Point pos(pOwner->p_main_node->pos_x,pOwner->p_main_node->pos_y);

    if(!EdgeCheck(pos,Point1,Point2))
    {
        if(AreaCheck(pos,area_a))
        {
            TaskCState::Instance()->task_c_fsm.ChangeState(TaskCAreaAState::Instance());
        }
        else
        {
            TaskCState::Instance()->task_c_fsm.ChangeState(TaskCAreaBState::Instance());
        }
    }
}

void TaskCAreaABState::Exit(TaskController *pOwner)
{
    
}

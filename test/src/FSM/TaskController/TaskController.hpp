#ifndef TASK_CONTROLLER_HPP
#define TASK_CONTROLLER_HPP

#include "StateMachine.hpp"
#include "../../main.hpp"
#include "TaskCFSM/TaskCAreaPoint.hpp"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Forward declaration
class TaskController;

class TaskFsm : public StateMachine<TaskController>
{
public:
    TaskFsm(TaskController *_pOwner) : StateMachine<TaskController>(_pOwner) {}

    void Init();
};

class TaskController
{
private:
    image_transport::ImageTransport* it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;

public:
    TaskFsm task_fsm;
    MainNode *p_main_node;

    TaskController();
    ~TaskController();

    void Init(MainNode *_p_main_node);
    void Update();

    void CVUpdate(const sensor_msgs::ImageConstPtr& msg);
};

#endif
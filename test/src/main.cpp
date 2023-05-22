#include "main.hpp"
#include "FSM/TaskController/TaskController.hpp"

MainNode* node = nullptr;

void PoseCallback(const geometry_msgs::Pose2D pos)
{	
	node->pos_theta = pos.theta;
    node->pos_x = pos.x;
    node->pos_y = pos.y;
}

void MainNode::UpdateSwitch()
{
    camera_switch_pub.publish(camera_switch);
    laser_switch_pub.publish(laser_switch);
}

void MainNode::UpdateParam()
{
    ros::param::get("/Task",task);
}

void MainNode::PubArea()
{
    
}

uint64_t count = 0;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "task_node");
    ros::NodeHandle n;

    MainNode main_node;

    node = &main_node;

    ros::Rate loop_rate(10);

    ros::Subscriber sub = n.subscribe("/pose2D",128,PoseCallback);

    main_node.pNodeHandle = &n;
    main_node.task_controller = new TaskController();

    main_node.task_controller->Init(&main_node);
    main_node.camera_switch_pub = n.advertise<std_msgs::Bool>("/vrep/camera_switch", 0);
    main_node.laser_switch_pub = n.advertise<std_msgs::Bool>("/vrep/laser_switch", 0);
    main_node.laser_switch.data = 0;
    main_node.camera_switch.data = 0;

    while (ros::ok())
    {
        main_node.UpdateParam();
        main_node.task_controller->Update();

        ros::spinOnce();
        loop_rate.sleep();

        if (count % 50 == 0)
        {
            main_node.UpdateSwitch();
        }

        count++;
    }

    delete main_node.task_controller;

    return 0;
}
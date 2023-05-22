#include <boost/thread/thread.hpp>
#include <termios.h>
#include <signal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/poll.h>

#include "TaskAState.hpp"

#define KEYCODE_W 0x77
#define KEYCODE_A 0x61
#define KEYCODE_S 0x73
#define KEYCODE_D 0x64

#define KEYCODE_A_CAP 0x41
#define KEYCODE_D_CAP 0x44
#define KEYCODE_S_CAP 0x53
#define KEYCODE_W_CAP 0x57

SmartCarKeyboardTeleopNode tbk;
boost::thread t;
int kfd = 0;
struct termios cooked, raw;
bool done;
bool *p_pub_flag;

void SmartCarKeyboardTeleopNode::Init(ros::NodeHandle *pn_)
{
    pn = pn_;

    pub_ = pn->advertise<geometry_msgs::Twist>("/vrep/cmd_vel", 1);

    ros::NodeHandle n_private("~");

    n_private.param("walk_vel", walk_vel_, 0.5);
    n_private.param("run_vel", run_vel_, 1.0);
    n_private.param("yaw_rate", yaw_rate_, 1.0);
    n_private.param("yaw_rate_run", yaw_rate_run_, 1.5);
}

void SmartCarKeyboardTeleopNode::keyboardLoop()
{
    char c;
    double max_tv = walk_vel_;
    double max_rv = yaw_rate_;
    bool dirty = false;
    int speed = 0;
    int turn = 0;

    // get the console in raw mode
    tcgetattr(kfd, &cooked);
    memcpy(&raw, &cooked, sizeof(struct termios));
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VEOL] = 1;
    raw.c_cc[VEOF] = 2;
    tcsetattr(kfd, TCSANOW, &raw);

    puts("Reading from keyboard");
    puts("Use WASD keys to control the robot");
    puts("Press Shift to move faster");

    struct pollfd ufd;
    ufd.fd = kfd;
    ufd.events = POLLIN;

    for (;;)
    {
        // get the next event from the keyboard
        int num;

        // puts("get the next event from the keyboard");

        if ((num = poll(&ufd, 1, 250)) < 0)
        {
            perror("poll():");
            return;
        }
        else if (num > 0)
        {
            if (read(kfd, &c, 1) < 0)
            {
                perror("read():");
                return;
            }
        }
        else
        {
            if (dirty == true)
            {
                stopRobot();
                dirty = false;
            }

            continue;
        }

        switch (c)
        {
        case KEYCODE_W:
            max_tv = walk_vel_;
            speed = 1;
            turn = 0;
            dirty = true;
            break;
        case KEYCODE_S:
            max_tv = walk_vel_;
            speed = -1;
            turn = 0;
            dirty = true;
            break;
        case KEYCODE_A:
            max_rv = yaw_rate_;
            speed = 0;
            turn = 1;
            dirty = true;
            break;
        case KEYCODE_D:
            max_rv = yaw_rate_;
            speed = 0;
            turn = -1;
            dirty = true;
            break;

        case KEYCODE_W_CAP:
            max_tv = run_vel_;
            speed = 1;
            turn = 0;
            dirty = true;
            break;
        case KEYCODE_S_CAP:
            max_tv = run_vel_;
            speed = -1;
            turn = 0;
            dirty = true;
            break;
        case KEYCODE_A_CAP:
            max_rv = yaw_rate_run_;
            speed = 0;
            turn = 1;
            dirty = true;
            break;
        case KEYCODE_D_CAP:
            max_rv = yaw_rate_run_;
            speed = 0;
            turn = -1;
            dirty = true;
            break;
        default:
            max_tv = walk_vel_;
            max_rv = yaw_rate_;
            speed = 0;
            turn = 0;
            dirty = false;
        }

        cmdvel_.linear.x = speed * max_tv;
        cmdvel_.angular.z = turn * max_rv;

        if ((*p_pub_flag) == true)
        {
            pub_.publish(cmdvel_);
        }
    }
}

void TaskAState::Init(TaskController *pOwner)
{
    p_pub_flag = &pub_flag;
    tbk.Init(pOwner->p_main_node->pNodeHandle);

    t = boost::thread(boost::bind(&SmartCarKeyboardTeleopNode::keyboardLoop, &tbk));
    t.interrupt();

    tcsetattr(kfd, TCSANOW, &cooked);
}

void TaskAState::Enter(TaskController *pOwner)
{
    pub_flag = true;
}

void TaskAState::Execute(TaskController *pOwner)
{
}

void TaskAState::Exit(TaskController *pOwner)
{
    pub_flag = false;
}

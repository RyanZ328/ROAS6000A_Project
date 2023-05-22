#include "TaskController.hpp"
#include "TaskAState.hpp"
#include "TaskBState.hpp"
#include "TaskCState.hpp"
#include "TaskDState.hpp"
#include "TaskEState.hpp"
#include "TaskNULLState.hpp"

void TaskFsm::Init()
{
    puts("TaskFsm::Init()");

    TaskAState::Instance()->Init(m_pOwner);
    TaskBState::Instance()->Init(m_pOwner);
    TaskCState::Instance()->Init(m_pOwner);
    TaskDState::Instance()->Init(m_pOwner);
    TaskEState::Instance()->Init(m_pOwner);

    SetCurrentState(TaskNULLState::Instance());
}

TaskController::TaskController() : task_fsm(this)
{
}

static const std::string OPENCV_WINDOW = "Image window";

TaskController::~TaskController()
{
    if (it_ != nullptr)
    {
        delete it_;
    }

    cv::destroyWindow(OPENCV_WINDOW);
}

void TaskController::Init(MainNode *_p_main_node)
{
    it_ = new image_transport::ImageTransport(*(_p_main_node->pNodeHandle));

    image_sub_ = it_->subscribe("/vrep/image", 1,
                                &TaskController::CVUpdate, this);
    image_pub_ = it_->advertise("/image_converter/output_video", 1);

    cv::namedWindow(OPENCV_WINDOW);

    p_main_node = _p_main_node;
    task_fsm.Init();
}

void TaskController::Update()
{
    task_fsm.Update();

    switch (p_main_node->task)
    {
    case 0:
        task_fsm.ChangeState(TaskNULLState::Instance());
        break;
    case 1:
        task_fsm.ChangeState(TaskAState::Instance());
        break;
    case 2:
        task_fsm.ChangeState(TaskBState::Instance());
        break;
    case 3:
        task_fsm.ChangeState(TaskCState::Instance());
        break;
    case 4:
        task_fsm.ChangeState(TaskDState::Instance());
        break;
    case 5:
        task_fsm.ChangeState(TaskEState::Instance());
        break;
    default:
        break;
    }
}

cv::Mat hsv;
cv::Mat hsvt;
cv::Mat hueMask, saturationMask, valueMask;
std::vector<cv::Mat> hsvChannel;

void TaskController::CVUpdate(const sensor_msgs::ImageConstPtr &msg)
{
    if (task_fsm.GetCurrentState() == TaskEState::Instance())
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

            cvtColor(cv_ptr->image, hsv, cv::COLOR_BGR2HSV);

            int x_size = 256;
            int y_size = 256;

            int x_mid[x_size] = {0};
            int x_points[x_size] = {0};
            int x_point = 0;
            int x_point_used_num = 0;

            int y_mid[y_size] = {0};

            int min_point = x_size / 16 + y_size / 16;
            int point_count = 0;
            int x_avg_bias = 0, x_avg = 0, x_sum = 0;
            int y_avg = 0, y_sum = 0;

            cv::Size dsize = cv::Size(x_size, y_size);
            resize(hsv, hsvt, dsize, 0, 0, cv::INTER_AREA);

            split(hsvt, hsvChannel);
            inRange(hsvChannel[0], 29, 32, hueMask);
            inRange(hsvChannel[1], 135, 255, saturationMask);
            inRange(hsvChannel[2], 120, 255, valueMask);
            hsvt = (hueMask & saturationMask) & valueMask;

            for (int i = 0; i < y_size; i++)
            {
                point_count = 0;
                x_sum = 0;
                for (int j = 0; j < x_size; j++)
                {
                    if (hsvt.data[i * y_size + j] == 255)
                    {
                        x_sum += j;
                        point_count++;
                    }
                }

                if (point_count > 4)
                {
                    x_point_used_num++;
                    x_mid[i] = x_sum / point_count;
                    x_points[i] = point_count;
                }
            }

            for (int i = 0; i < y_size; i++)
            {
                if (x_mid[i] != 0)
                {
                    x_avg += x_mid[i] * x_points[i];
                    x_point += x_points[i];
                }
            }

            if (x_point != 0)
            {
                x_avg /= x_point;
            }

            double pointd = (double)x_point;
            double dis = 2.85168 + 1734.53 / pointd - 0.000294313 * pointd + 0.0000000106784 * pointd * pointd;

            // std::cout<<"x:"<<hsvt<<std::endl;
            std::cout << "x_avg:" << ((float)x_avg / 256.0f) << std::endl;
            std::cout << "dis:" << dis << std::endl;
            std::cout << "point_num:" << x_point << std::endl;
        }
        catch (cv_bridge::Exception &e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // Draw an example circle on the video stream

        // Update GUI Window
        cv::imshow(OPENCV_WINDOW, cv_ptr->image);

        // cv::imshow("HSV",hsv);
        cv::imshow("HSV", hsvt);
        cv::waitKey(10);

        // Output modified video stream
        image_pub_.publish(cv_ptr->toImageMsg());
    }
}
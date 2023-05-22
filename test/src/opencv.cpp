 #include <ros/ros.h>
 #include <image_transport/image_transport.h>
 #include <cv_bridge/cv_bridge.h>
 #include <sensor_msgs/image_encodings.h>
 #include <opencv2/imgproc/imgproc.hpp>
 #include <opencv2/highgui/highgui.hpp>
  static const std::string OPENCV_WINDOW = "Image window";
 
 class ImageConverter
 {
   ros::NodeHandle nh_;
   image_transport::ImageTransport it_;
   image_transport::Subscriber image_sub_;
   image_transport::Publisher image_pub_;
 
 public:
   ImageConverter()
     : it_(nh_)
   {
     // Subscrive to input video feed and publish output video feed
     image_sub_ = it_.subscribe("/vrep/image", 1,
       &ImageConverter::imageCb, this);
     image_pub_ = it_.advertise("/image_converter/output_video", 1);

    cv::namedWindow(OPENCV_WINDOW);
   }
 
   ~ImageConverter()
   {
     cv::destroyWindow(OPENCV_WINDOW);
   }
 
    cv::Mat hsv;
    cv::Mat hsvt;
    cv::Mat hueMask,saturationMask ,valueMask;
    std::vector<cv::Mat> hsvChannel;
   void imageCb(const sensor_msgs::ImageConstPtr& msg)
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

        int min_point = x_size/16 + y_size/16;
        int point_count = 0;
        int x_avg_bias = 0,x_avg = 0,x_sum = 0;
        int y_avg = 0,y_sum = 0;

        cv::Size dsize = cv::Size(x_size,y_size);
        resize(hsv,hsvt,dsize,0,0,cv::INTER_AREA);


        split(hsvt, hsvChannel);
        inRange(hsvChannel[0],29,32,hueMask);
        inRange(hsvChannel[1],135,255,saturationMask);
        inRange(hsvChannel[2],120,255,valueMask);
        hsvt = (hueMask & saturationMask) & valueMask;

        for(int i = 0; i < y_size; i++)
        {
            point_count = 0;
            x_sum = 0;
            for(int j = 0; j < x_size; j++)
            {
                if(hsvt.data[i*y_size+j]==255)
                {
                    x_sum+=j;
                    point_count++;
                }
            }

            if(point_count>4)
            {
                x_point_used_num ++;
                x_mid[i] = x_sum / point_count;
                x_points[i] = point_count;
            }
        }

        for(int i = 0; i < y_size; i++)
        {
            if(x_mid[i]!=0)
            {
                x_avg+=x_mid[i]*x_points[i];
                x_point+=x_points[i];
            }
        }

        if(x_point!=0)
        {
            x_avg/=x_point;
        }
        
        double pointd = (double) x_point;
        double dis = 2.85168 + 1734.53/pointd -0.000294313*pointd + 0.0000000106784 * pointd * pointd;

        //std::cout<<"x:"<<hsvt<<std::endl;    
        std::cout<<"x_avg:"<<((float)x_avg/256.0f)<<std::endl;
        std::cout<<"dis:"<<dis<<std::endl;
        std::cout<<"point_num:"<<x_point<<std::endl;
     }
     catch (cv_bridge::Exception& e)
     {
       ROS_ERROR("cv_bridge exception: %s", e.what());
       return;
     }

     // Draw an example circle on the video stream

     // Update GUI Window
    cv::imshow(OPENCV_WINDOW, cv_ptr->image);

	//cv::imshow("HSV",hsv);
    cv::imshow("HSV",hsvt);
    cv::waitKey(10);
 
     // Output modified video stream
     image_pub_.publish(cv_ptr->toImageMsg());
   }
 };
 
 int main(int argc, char** argv)
 {
   ros::init(argc, argv, "image_converter");
   ImageConverter ic;
   ros::spin();
   return 0;
 }
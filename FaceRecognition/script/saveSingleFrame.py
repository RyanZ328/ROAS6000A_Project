import os
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Int8
import time

COLOR_FRAME_TOPIC = '/vrep/image'

class recognition:
    def __init__(self):
        self.imgSub = rospy.Subscriber(COLOR_FRAME_TOPIC, Image, self.image_receive, queue_size=1)
        self.imgPosePub = rospy.Publisher('Location', Int8, queue_size=1)
        self.bridge = CvBridge()
        self.color_image = None
        self.rate = rospy.Rate(10)
        self.flagImageSub = False
        self.face_cascade = cv2.CascadeClassifier(os.path.split(os.path.realpath(__file__))[0]+'/haarcascade_frontalface_default.xml')
        self.Loop()
        
        
    def Loop(self):
        while(True):
            if (self.flagImageSub):
                color_img = cv2.flip(self.color_image,1)
                cv2.imshow("frame",color_img)
                gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2)
                print("Detection:",faces)
                color_frame_cp = color_img.copy()
                time.sleep(0.2)
                bestx = -1
                besty = -1
                highestsize = 0
                detected = False
                for (x,y,w,h) in faces:
                    # 在原彩色图上画人脸矩形框
                    detected = True
                    cv2.rectangle(color_frame_cp,(x,y),(x+w,y+h),(255,255,0),2)
                    size = w*h
                    if size > highestsize:
                        bestx = x + 0.5*w
                        besty = y + 0.5*h
                # # 显示画好矩形框的图片
                
                if detected:
                    messsage = Int8()
                    angle = int(bestx-256*45/512)
                    print('Angle:',angle)
                    messsage.data = angle
                    self.imgPosePub.publish(messsage)
                
                cv2.imshow('faces',color_frame_cp)
                cv2.imwrite("cap.jpg", color_img)
                
                key = cv2.waitKey(100)
                exit(0)
                
                # break
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    exit(0)
                    break
    
    def image_receive(self, data):
        try:
            color_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.color_image = color_image.copy()
            # rospy.loginfo("Frame recived")
            self.flagImageSub = True
        except CvBridgeError as e:
            print(e)

        
if __name__ == '__main__':
    try:
        rospy.init_node('recognition', anonymous=False)
        rospy.loginfo("Init recognition main")   
        recognition()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("End recognition main")
        
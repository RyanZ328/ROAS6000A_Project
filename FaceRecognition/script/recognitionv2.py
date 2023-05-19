#! /usr/bin/env python3

import os
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Int8
import time
import animeface
from PIL import Image as P_Image

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
        self.anime_face_cascade = cv2.CascadeClassifier(os.path.split(os.path.realpath(__file__))[0]+'/lbpcascade_animeface.xml')
        
        self.imageRef1 = cv2.imread(os.path.split(os.path.realpath(__file__))[0]+'/1.jpg',cv2.IMREAD_UNCHANGED)
        self.imageRef2 = cv2.imread(os.path.split(os.path.realpath(__file__))[0]+'/2.jpg',cv2.IMREAD_UNCHANGED)
        self.imageRef3 = cv2.imread(os.path.split(os.path.realpath(__file__))[0]+'/3.jpg',cv2.IMREAD_UNCHANGED)
        self.imageRef4 = cv2.imread(os.path.split(os.path.realpath(__file__))[0]+'/4.jpg',cv2.IMREAD_UNCHANGED)
        self.imageRef5 = cv2.imread(os.path.split(os.path.realpath(__file__))[0]+'/5.jpg',cv2.IMREAD_UNCHANGED)
        self.imageRefGray1 = cv2.cvtColor(self.imageRef1, cv2.COLOR_BGR2GRAY)
        self.imageRefGray2 = cv2.cvtColor(self.imageRef2, cv2.COLOR_BGR2GRAY)
        self.imageRefGray3 = cv2.cvtColor(self.imageRef3, cv2.COLOR_BGR2GRAY)
        self.imageRefGray4 = cv2.cvtColor(self.imageRef4, cv2.COLOR_BGR2GRAY)
        self.imageRefGray5 = cv2.cvtColor(self.imageRef5, cv2.COLOR_BGR2GRAY)
        self.imageRefList = [self.imageRef1,self.imageRef2,self.imageRef3,self.imageRef4,self.imageRef5]
        self.imageRefGrayList = [self.imageRefGray1,self.imageRefGray2,self.imageRefGray3,self.imageRefGray4,self.imageRefGray5]
        self.describerList = []
        
        self.sift = cv2.SIFT_create()
        self.bfm = cv2.BFMatcher()
        
        for i in range(5):
            kp1,des1 = self.sift.detectAndCompute(self.imageRefGrayList[i],None)
            self.describerList.append(des1)
            
        self.names = ["Obama","Aneme","Avril","ChineseMan","WhiteMan"]
        
        # img = self.imageRef5.copy()
        # img=cv2.drawKeypoints(self.imageRefGray5,kp1,img)
        
        
        # self.imageRef5s = cv2.imread(os.path.split(os.path.realpath(__file__))[0]+'/5si.jpg',cv2.IMREAD_UNCHANGED)
        # self.imageRefGray5s = cv2.cvtColor(self.imageRef5s, cv2.COLOR_BGR2GRAY)
        
        # kp2,des2 = self.sift.detectAndCompute(self.imageRefGray5s,None)
        # img2 = self.imageRef5s.copy()
        # img2=cv2.drawKeypoints(self.imageRefGray5s,kp1,img2)
        
        
        # match = self.bfm.match(des1, des2)
        # img_match1 = cv2.drawMatches(self.imageRef5, kp1, self.imageRef5s, kp2, match, None)
        # cv2.imshow("match2",img_match1)
        # distancetotal = 0
        # for mth in match:
        #     # print(mth.distance)
        #     distancetotal = distancetotal + mth.distance
        # print(distancetotal)
        # print(distancetotal/len(match))
        # print(len(match))
        
        # # 2nd
        # kp1,des1 = self.sift.detectAndCompute(self.imageRefGray2,None)
        # img = self.imageRef2.copy()
        # img=cv2.drawKeypoints(self.imageRefGray2,kp1,img)
        
        
        # self.imageRef5s = cv2.imread(os.path.split(os.path.realpath(__file__))[0]+'/5si.jpg',cv2.IMREAD_UNCHANGED)
        # self.imageRefGray5s = cv2.cvtColor(self.imageRef5s, cv2.COLOR_BGR2GRAY)
        
        # kp2,des2 = self.sift.detectAndCompute(self.imageRefGray5s,None)
        # img2 = self.imageRef5s.copy()
        # img2=cv2.drawKeypoints(self.imageRefGray5s,kp1,img2)
        
        
        # match = self.bfm.match(des1, des2)
        # img_match1 = cv2.drawMatches(self.imageRef2, kp1, self.imageRef5s, kp2, match, None)
        # cv2.imshow("match",img_match1)
        # # print(match)
        # distancetotal = 0
        # for mth in match:
        #     # print(mth.distance)
        #     distancetotal = distancetotal + mth.distance
        # print(distancetotal)
        # print(distancetotal/len(match))
        # print(len(match))
        
        # cv2.imwrite('sift_keypoints.jpg',img)
        # cv2.waitKey(2000)
        
        self.Loop()
        
        
    def Loop(self):
        while(True):
            if (self.flagImageSub):
                color_img = cv2.flip(self.color_image,1)
                # cv2.imshow("frame",color_img)
                gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
                anm_faces = self.anime_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
                # histeq_gray = cv2.equalizeHist(gray)
                anm_faces2 = animeface.detect(P_Image.fromarray(gray))
                # print("Detection:",faces)
                color_frame_cp = color_img.copy()
                time.sleep(0.2)
                bestx = -1
                besty = -1
                highestsize = 0
                detected = False
                for each in anm_faces2:  # 遍历所有检测到的动漫脸
                    temp = each.face.pos
                    x = temp.x
                    y = temp.y
                    w = temp.width
                    h = temp.height
                    cv2.rectangle(color_frame_cp, (x, y), (x+w, y+h), (0, 0, 255), 2)  # 绘制矩形框
                for (x,y,w,h) in anm_faces:
                    # 在原彩色图上画人脸矩形框
                    detected = True
                    cv2.rectangle(color_frame_cp,(x,y),(x+w,y+h),(255,0,0),2)
                    size = w*h
                    if size > highestsize:
                        bestx = x + 0.5*w
                        besty = y + 0.5*h
                for (x,y,w,h) in faces:
                    # 在原彩色图上画人脸矩形框
                    detected = True
                    cv2.rectangle(color_frame_cp,(x,y),(x+w,y+h),(0,255,0),2)
                    size = w*h
                    if size > highestsize:
                        bestx = x + 0.5*w
                        besty = y + 0.5*h
                # 显示画好矩形框的图片
                
                if detected:
                    messsage = Int8()
                    angle = int((bestx-256)*45/512)
                    # print('Angle:',angle)
                    messsage.data = angle
                    self.imgPosePub.publish(messsage)
                
                kp2,des2 = self.sift.detectAndCompute(gray,None)
                SimilarID = -1
                if des2 is not None:
                    leastDistance = 1000
                    for i in range(5):
                        desTemplate = self.describerList[i]
                        match = self.bfm.match(desTemplate, des2)
                        distancetotal = 0
                        for mth in match:
                            # print(mth.distance)
                            distancetotal = distancetotal + mth.distance
                        # print(distancetotal)
                        distanceTemp = distancetotal/len(match)
                        if distanceTemp < leastDistance:
                            SimilarID = i
                            leastDistance = distanceTemp
                        print("Distance: " + str(distancetotal/len(match)))
                        # print(len(match))
                    if (SimilarID!=-1):
                        print(self.names[SimilarID])
                    
                    
                
                cv2.imshow('faces',color_frame_cp)
                key = cv2.waitKey(50)
                
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
        
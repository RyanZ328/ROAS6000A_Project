#! /usr/bin/env python3

import os
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image,LaserScan
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Int8
import time
import math
# import animeface
# from PIL import Image as P_Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product as product
from math import ceil
# import sys
import tf

COLOR_FRAME_TOPIC = '/vrep/image'
LIDAR_TOPIC = '/vrep/scan'

cfg = {
    'name': 'FaceBoxes',
    #'min_dim': 1024,
    #'feature_maps': [[32, 32], [16, 16], [8, 8]],
    # 'aspect_ratios': [[1], [1], [1]],
    'min_sizes': [[32, 64, 128], [256], [512]],
    'steps': [32, 64, 128],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True
}



cpu = True
# confidenceTh = 0.05
confidenceTh = 0.1
nmsTh = 0.3
keepTopK = 750
top_k = 5000

model_path = os.path.split(os.path.realpath(__file__))[0]+'/ssd_anime_face_detect.pth'

class recognition:
    def __init__(self):

        torch.set_grad_enabled(False)
        # net and model
        # initialize detector
        self.net = FaceBoxes(phase='test', size=None, num_classes=2)
        self.net = load_model(self.net, model_path, cpu)
        self.net.eval()
        #print('Finished loading model!')
        #print(net)
        self.device = torch.device("cpu" if cpu else "cuda")
        self.net = self.net.to(self.device)
        self.lidarInitialized = False
        self.lidarScan = LaserScan()

        
        
        self.imgSub = rospy.Subscriber(COLOR_FRAME_TOPIC, Image, self.image_receive, queue_size=1)
        self.lidarSub = rospy.Subscriber(LIDAR_TOPIC, LaserScan , self.scan_receive, queue_size=1)
        # self.imgPosePub = rospy.Publisher('Location', Marker, queue_size=1)
        self.imgTfBroadcaster = tf.TransformBroadcaster()
        # datas.pose.position.z = 0.1
        # datas.pose.position.x = 0
        self.bridge = CvBridge()
        self.color_image = None
        self.rate = rospy.Rate(10)
        self.imageInitialized = False
        self.face_cascade = cv2.CascadeClassifier(os.path.split(os.path.realpath(__file__))[0]+'/haarcascade_frontalface_default.xml')
        # self.anime_face_cascade = cv2.CascadeClassifier(os.path.split(os.path.realpath(__file__))[0]+'/lbpcascade_animeface.xml')
        
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
            
        self.names = ["Obama","Anime","Avril","ChineseMan","WhiteMan"]
        
        self.Loop()
        
        
    def Loop(self):
        
        last_Detected = False
        
        while(True):
            if (self.imageInitialized & self.lidarInitialized):
                color_img = cv2.flip(self.color_image,1)
                # cv2.imshow("frame",color_img)
                gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
                
                color_frame_cp = color_img.copy()
                time.sleep(0.2)
                bestx = -1
                besty = -1
                highestsize = 0
                detected = False
                an_detected = False
                
                # torch model anime face recognition
                dets = ssd_detect_anime(color_img,self.net,self.device)
                for k in range(dets.shape[0]):
                    an_detected = True
                    xmin = dets[k, 0]
                    ymin = dets[k, 1]
                    xmax = dets[k, 2]
                    ymax = dets[k, 3]
                    ymin += 0.2 * (ymax - ymin + 1)
                    score = dets[k, 4]
                    # print('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(image_path, score, xmin, ymin, xmax, ymax))
                    cv2.rectangle(color_frame_cp, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 255), 2)
                    size = abs((xmax-xmin)*(ymax-ymin))
                    if size > highestsize:
                        bestx = (xmax+xmin)/2
                
                
                # abandon face recog method
                # anm_faces2 = animeface.detect(P_Image.fromarray(gray))
                # for each in anm_faces2:  # 遍历所有检测到的动漫脸
                #     temp = each.face.pos
                #     x = temp.x
                #     y = temp.y
                #     w = temp.width
                #     h = temp.height
                #     cv2.rectangle(color_frame_cp, (x, y), (x+w, y+h), (0, 0, 255), 2)  # 绘制矩形框
                
                # abandon face recog method
                # anm_faces = self.anime_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
                # for (x,y,w,h) in anm_faces:
                #     # 在原彩色图上画人脸矩形框
                #     detected = True
                #     cv2.rectangle(color_frame_cp,(x,y),(x+w,y+h),(255,0,0),2)
                #     size = w*h
                #     if size > highestsize:
                #         bestx = x + 0.5*w
                #         besty = y + 0.5*h
                
                # standard cascade classifier face recognition model
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                for (x,y,w,h) in faces:
                    # 在原彩色图上画人脸矩形框
                    detected = True
                    cv2.rectangle(color_frame_cp,(x,y),(x+w,y+h),(0,255,0),2)
                    size = w*h
                    if size > highestsize:
                        bestx = x + 0.5*w
                        besty = y + 0.5*h
                # 显示画好矩形框的图片
                
                if (detected or an_detected):
                    last_Detected = True
                    
                    # determine the distance using lidar info after knowing image angle w.r.t. robot
                    angle = (bestx-256.0)*45.0/512.0
                    index = int(len(self.lidarScan.ranges)/2 + angle/abs(self.lidarScan.angle_increment*180/3.1415))
                    
                    if index<0:
                        index = 0
                    if index > len(self.lidarScan.ranges)-1:
                        index = len(self.lidarScan.ranges)-1
                    imgDistance = self.lidarScan.ranges[index]
                    anglePi = angle*3.1416/180
                    xdistance = imgDistance*math.cos(anglePi)
                    ydistance = -imgDistance*math.sin(anglePi)
                    self.imgTfBroadcaster.sendTransform((xdistance, ydistance, 0),
                                tf.transformations.quaternion_from_euler(0, 0, 0),
                                rospy.Time.now(),
                                "image_pose",
                                "base_link")
                    
                    # compare image with SIFT features
                    kp2,des2 = self.sift.detectAndCompute(gray,None)
                    SimilarID = -1
                    if des2 is not None:
                        leastDistance = 1000
                        for i in range(5):
                            desTemplate = self.describerList[i]
                            match = self.bfm.match(desTemplate, des2)
                            distancetotal = 0
                            for mth in match:
                                distancetotal = distancetotal + mth.distance
                            distanceTemp = distancetotal/len(match)
                            if distanceTemp < leastDistance:
                                SimilarID = i
                                leastDistance = distanceTemp
                        if (SimilarID!=-1):
                            print(self.names[SimilarID])
                else:
                    if (last_Detected):
                        # TODO Update world frame image posiiton transformation
                        pass
                    last_Detected = False
                    
                    
                    
                
                cv2.imshow('faces',color_frame_cp)
                key = cv2.waitKey(50)
                
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    exit(0)
                    break
            else:
                time.sleep(0.1)
                
    def scan_receive(self,data):
        try:
            self.lidarScan = data
            self.lidarInitialized = True
        except CvBridgeError as e:
            print(e)
        
    
    def image_receive(self, data):
        try:
            color_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.color_image = color_image.copy()
            self.imageInitialized = True
        except CvBridgeError as e:
            print(e)



# Below are torch codes used for anime face recognition


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):

  def __init__(self):
    super(Inception, self).__init__()
    self.branch1x1 = BasicConv2d(128, 32, kernel_size=1, padding=0)
    self.branch1x1_2 = BasicConv2d(128, 32, kernel_size=1, padding=0)
    self.branch3x3_reduce = BasicConv2d(128, 24, kernel_size=1, padding=0)
    self.branch3x3 = BasicConv2d(24, 32, kernel_size=3, padding=1)
    self.branch3x3_reduce_2 = BasicConv2d(128, 24, kernel_size=1, padding=0)
    self.branch3x3_2 = BasicConv2d(24, 32, kernel_size=3, padding=1)
    self.branch3x3_3 = BasicConv2d(32, 32, kernel_size=3, padding=1)

  def forward(self, x):
    branch1x1 = self.branch1x1(x)

    branch1x1_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch1x1_2 = self.branch1x1_2(branch1x1_pool)

    branch3x3_reduce = self.branch3x3_reduce(x)
    branch3x3 = self.branch3x3(branch3x3_reduce)

    branch3x3_reduce_2 = self.branch3x3_reduce_2(x)
    branch3x3_2 = self.branch3x3_2(branch3x3_reduce_2)
    branch3x3_3 = self.branch3x3_3(branch3x3_2)

    outputs = [branch1x1, branch1x1_2, branch3x3, branch3x3_3]
    return torch.cat(outputs, 1)


class CRelu(nn.Module):

  def __init__(self, in_channels, out_channels, **kwargs):
    super(CRelu, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
    self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = torch.cat([x, -x], 1)
    x = F.relu(x, inplace=True)
    return x


class FaceBoxes(nn.Module):

  def __init__(self, phase, size, num_classes):
    super(FaceBoxes, self).__init__()
    self.phase = phase
    self.num_classes = num_classes
    self.size = size

    self.conv1 = CRelu(3, 24, kernel_size=7, stride=4, padding=3)
    self.conv2 = CRelu(48, 64, kernel_size=5, stride=2, padding=2)

    self.inception1 = Inception()
    self.inception2 = Inception()
    self.inception3 = Inception()

    self.conv3_1 = BasicConv2d(128, 128, kernel_size=1, stride=1, padding=0)
    self.conv3_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)

    self.conv4_1 = BasicConv2d(256, 128, kernel_size=1, stride=1, padding=0)
    self.conv4_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)

    self.loc, self.conf = self.multibox(self.num_classes)

    if self.phase == 'test':
        self.softmax = nn.Softmax(dim=-1)

    if self.phase == 'train':
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

  def multibox(self, num_classes):
    loc_layers = []
    conf_layers = []
    loc_layers += [nn.Conv2d(128, 21 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(128, 21 * num_classes, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
    return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

  def forward(self, x):

    detection_sources = list()
    loc = list()
    conf = list()

    x = self.conv1(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    x = self.conv2(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    x = self.inception1(x)
    x = self.inception2(x)
    x = self.inception3(x)
    detection_sources.append(x)

    x = self.conv3_1(x)
    x = self.conv3_2(x)
    detection_sources.append(x)

    x = self.conv4_1(x)
    x = self.conv4_2(x)
    detection_sources.append(x)

    for (x, l, c) in zip(detection_sources, self.loc, self.conf):
        loc.append(l(x).permute(0, 2, 3, 1).contiguous())
        conf.append(c(x).permute(0, 2, 3, 1).contiguous())

    loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
    conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

    if self.phase == "test":
      output = (loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(-1, self.num_classes)))
    else:
      output = (loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes))

    return output


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        #self.aspect_ratios = cfg['aspect_ratios']
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [
            [ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    if min_size == 32:
                        dense_cx = [x*self.steps[k]/self.image_size[1]
                            for x in [j+0, j+0.25, j+0.5, j+0.75]]
                        dense_cy = [y*self.steps[k]/self.image_size[0]
                            for y in [i+0, i+0.25, i+0.5, i+0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    elif min_size == 64:
                        dense_cx = [x*self.steps[k]/self.image_size[1]
                            for x in [j+0, j+0.5]]
                        dense_cy = [y*self.steps[k]/self.image_size[0]
                            for y in [i+0, i+0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    else:
                        cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                        cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                        anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def mymax(a, b):
    if a >= b:
        return a
    else:
        return b


def mymin(a, b):
    if a >= b:
        return b
    else:
        return a


def cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=int)
    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = mymax(ix1, x1[j])
            yy1 = mymax(iy1, y1[j])
            xx2 = mymin(ix2, x2[j])
            yy2 = mymin(iy2, y2[j])
            w = mymax(0.0, xx2 - xx1 + 1)
            h = mymax(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1
    return keep


def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if force_cpu:
        #return cpu_soft_nms(dets, thresh, method = 0)
        return cpu_nms(dets, thresh)
    return cpu_nms(dets, thresh)


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    # print('remove prefix \'{}\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    # print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def ssd_detect_anime(imgOrig,net,device):
    img = np.float32(imgOrig)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor(
        [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    loc, conf = net(img)  # forward pass
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.data.cpu().numpy()[:, 1]

    # ignore low scores
    inds = np.where(scores > confidenceTh)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
        np.float32, copy=False)
    #keep = py_cpu_nms(dets, args.nms_threshold)
    keep = nms(dets, nmsTh, force_cpu=cpu)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:keepTopK, :]
    return dets

if __name__ == '__main__':
    try:
        rospy.init_node('recognition', anonymous=False)
        rospy.loginfo("Init recognition main")   
        recognition()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("End recognition main")
        
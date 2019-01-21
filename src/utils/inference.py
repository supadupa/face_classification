import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
# torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime,time,math
import argparse
import numpy as np

from SFD_pytorch.net_s3fd import s3fd
from SFD_pytorch.bbox import *

def load_image(image_path, grayscale=False, target_size=None):
    pil_image = image.load_img(image_path, grayscale, target_size)
    return image.img_to_array(pil_image)

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

def load_s3fd_model(model_path):
    net = s3fd()
    net.load_state_dict(torch.load(model_path))
    net.eval()
    return net

def detect_faces_sfd(detection_model, gray_image_array):
    img = gray_image_array - np.array([104,117,123])
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,)+img.shape)

    with torch.no_grad():
        img = Variable(torch.from_numpy(img).float())# .cuda()
        BB,CC,HH,WW = img.size()
        olist = detection_model(img)

    bboxlist = []
    for i in range(int(len(olist)/2)): olist[i*2] = F.softmax(olist[i*2])
    for i in range(int(len(olist)/2)):
        ocls,oreg = olist[i*2].data.cpu(),olist[i*2+1].data.cpu()
        FB,FC,FH,FW = ocls.size() # feature map size
        stride = 2**(i+2)    # 4,8,16,32,64,128
        anchor = stride*4
        for Findex in range(FH*FW):
            windex,hindex = Findex%FW,Findex//FW
            axc,ayc = stride/2+windex*stride,stride/2+hindex*stride
            score = ocls[0,1,hindex,windex]
            loc = oreg[0,:,hindex,windex].contiguous().view(1,4)
            if score<0.05: continue
            priors = torch.Tensor([[axc/1.0,ayc/1.0,stride*4/1.0,stride*4/1.0]])
            variances = [0.1,0.2]
            box = decode(loc,priors,variances)
            x1,y1,x2,y2 = box[0]*1.0
            # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
            bboxlist.append([x1,y1,x2,y2,score])
    bboxlist = np.array(bboxlist)
    if 0==len(bboxlist): bboxlist=np.zeros((1, 5))
    keep = nms(bboxlist,0.3)
    bboxlist = bboxlist[keep,:]

    boxes = []
    for b in bboxlist:
        x1,y1,x2,y2,s = b
        if s<0.5: continue
        
        boxes.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
 
    return boxes 

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

def get_colors(num_classes):
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    colors = np.asarray(colors) * 255
    return colors


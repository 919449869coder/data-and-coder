# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:08:03 2021

@author: Admin
"""


import os
import numpy as np 
import cv2
from skimage import feature as ft 
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import copy
from timeit import default_timer as timer
from skimage.feature import greycomatrix, greycoprops
from sklearn import preprocessing
from skimage import morphology,measure

from PIL import Image 


def watershed_show_obj(GrayImage,img,color):
    
  ret,img_bin=cv2.threshold(GrayImage,color,255, cv2.THRESH_BINARY_INV)  #254
    
  kernel = np.ones((3,3),np.uint8)
  opening = cv2.morphologyEx(img_bin,cv2.MORPH_OPEN,kernel, iterations =2)
# sure background area   膨胀找背景
  sure_bg = cv2.dilate(opening,kernel,iterations=5)
# Finding sure foreground area  距离变换  在二值化  找到啊种子中心附近  作为前景
  dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
  ret, sure_fg = cv2.threshold(dist_transform,0.6*dist_transform.max(),255,0)
# Finding unknown region    背景与前景相减 得到我未知区域
  sure_fg = np.uint8(sure_fg)
  unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling     前景加上标签
  ret, markers = cv2.connectedComponents(sure_fg)    
# Add one to all labels so that sure background is not 0, but 1
  markers = markers+1   
# Now, mark the region of unknown with zero
  markers[unknown==255] = 0
  markers = cv2.watershed(img,markers)
  img[markers == -1] = [255,0,0]
  
  return markers,ret


def get_img_size(markers):
  sp = markers.shape 
  height = sp[0]
  width = sp[1]
  return height,width

def get_all_class_coordinate(ret,height,width,markers):
  ret = ret-1
  area = [[] for x in range(ret)]
  for i in range(height):
      for j in range(width):                   
          I = markers[i,j]
          if (I>=2):
              c = [i,j]
              area[I-2].append(c)                    
  return area
  
def obj_corner_coordinate(area):
  p = len(area) 
  e = [[] for x in range(p)]
  rects = [[] for x in range(p)]
  rectss = [[] for x in range(p)]
  for t in range(p):       
    d = np.array(area[t])
    x_max=d[:,0].max()
    x_min=d[:,0].min()
    y_max=d[:,1].max()
    y_min=d[:,1].min()
        
    y=x_min
    x=y_min
    h=x_max-x_min
    w=y_max-y_min
    
    
    rects[t]=[x,y,w,h] #此tects用于画框
    
  return rects,p
#加白边
def jiabaibian(proposal):
    #old_im = Image.open('C:/Users/Admin/Desktop/img/1.jpg')
    old_im = Image.fromarray(cv2.cvtColor(proposal,cv2.COLOR_BGR2RGB)) 
    #old_im = Image.open('C:/Users/Admin/Desktop/img/1.jpg')
    old_size = old_im.size
    #print(old_size)
    
    new_size = (250, 250)
    #print(new_size)
    new_im = Image.new("RGB", new_size,"black")   ## luckily, this is already black!
    x = int((new_size[0]-old_size[0])/2)
    y = int((new_size[1]-old_size[1])/2)
    new_im.paste(old_im, (x,y))
    
    return new_im

test_dir = "C:/Users/Admin/Desktop/img/rgb/"
test_dir_nir = "C:/Users/Admin/Desktop/img/nir/"

save_dir = "C:/Users/Admin/Desktop/img/rgb_fen/"
save_dir_nir = "C:/Users/Admin/Desktop/img/nir_fen/"
save_dir_rgb_nir = "C:/Users/Admin/Desktop/img/rgb_nir/"

# img = cv2.imread("C:/Users/Admin/Desktop/img/rgb/good_1.jpg")
# img_nir = cv2.imread("C:/Users/Admin/Desktop/img/nir/good_3.jpg")
# img_rgb_nir = cv2.imread('rgb_nir.jpg')
# img_rgb_nir1 = cv2.imread('rgb_nir.png')
#plt.imshow(markers_nir)

img_names = os.listdir(test_dir)
img_names_nir = os.listdir(test_dir_nir)


for img_name in img_names:
    print(img_name)
    img = cv2.imread(os.path.join(test_dir, img_name))
    GrayImage = img[:,:,0]
    
    img_nir = cv2.imread(os.path.join(test_dir_nir, img_name))
    GrayImage_nir=cv2.cvtColor(img_nir,cv2.COLOR_BGR2GRAY) 

    markers,ret=watershed_show_obj(GrayImage,img,color=210)

    height,width=get_img_size(markers)
    area=get_all_class_coordinate(ret,height,width,markers)
    rects,p=obj_corner_coordinate(area)

    rows, cols, _ = img.shape
    
    for rect in rects:
        xc = int(rect[0] + rect[2]/2)
        yc = int(rect[1] + rect[3]/2)
    
        size = max(rect[2], rect[3])
        x1 = max(0, int(xc-size/2))
        y1 = max(0, int(yc-size/2))
        x2 = min(cols, int(xc+size/2))
        y2 = min(rows, int(yc+size/2))
        
        newxmin=rect[1] #上
        newymin=rect[0]
        newxmax=rect[3]+rect[1]
        newymax=rect[2]+rect[0]
        
        rectss=[newxmin,newymin,newxmax,newymax]


        m=rects.index(rect)+2                         
        oneimg = copy.deepcopy(img)
        oneimg_nir = copy.deepcopy(img_nir)
        # oneimg[markers != m] = [255,255,255]      #判断像素点值不是标签值得变为o
        # oneimg_nir[markers_nir != m] = [255,255,255]      #判断像素点值不是标签值得变为o
        
        oneimg[markers != m] = [0,0,0]      #判断像素点值不是标签值得变为o
        oneimg_nir[markers != m] = [0,0,0]      #判断像素点值不是标签值得变为o
        
        x_min,y_min,x_max,y_max = rectss        
        
        proposal = oneimg[x_min:x_max,y_min:y_max]
        proposal_nir = oneimg_nir[x_min:x_max,y_min:y_max]
        

        proposal = jiabaibian(proposal)
        proposal_nir = jiabaibian(proposal_nir)
        proposal = cv2.cvtColor(np.asarray(proposal),cv2.COLOR_RGB2BGR) 
        proposal_nir = cv2.cvtColor(np.asarray(proposal_nir),cv2.COLOR_RGB2BGR) 
        
        pnir=proposal_nir[:,:,1]

        p_rgb_nir = cv2.merge([proposal,pnir])
        
        #写一个.txt文件夹 内容为1，每运行一次加一
        f_out = open('C:/Users/Admin/Desktop/img/1.txt', 'r+')
        a = f_out.read()
        a = int(a) + 1
        print(a)
        f_out.seek(0)
        f_out.truncate()
        f_out.write(str(a))
        f_out.close()

        #i = x_min+x_max+y_max+y_min
        cv2.imwrite(save_dir + "/" + '%d.jpg'%        a, proposal)
        cv2.imwrite(save_dir_nir + "/" + '%d.jpg'%a, proposal_nir)
        cv2.imwrite(save_dir_rgb_nir + "/" + '%d.png'%a, p_rgb_nir)
        #img_rgb_ni1 = cv2.imread('rgb_ni.png',cv2.IMREAD_UNCHANGED)  读取四通道png
   # plt.imshow(rectss)
#写一个.txt文件夹 内容为1，每运行一次加一
f_out = open('C:/Users/Admin/Desktop/img/1.txt', 'r+')
a = f_out.read()
a = 0
f_out.seek(0)
f_out.truncate()
f_out.write(str(a))
f_out.close()


    
    
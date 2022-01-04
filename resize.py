# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:55:03 2021

@author: Admin
"""

#白边变黑边
import os
import cv2
from PIL import Image 
import numpy as np 
from matplotlib import pyplot as plt

#加白边
def jiabaibian(t):
    #old_im = Image.open('C:/Users/Admin/Desktop/img/1.jpg')
   ## Image.fromarray(cv2.cvtColor(proposal,cv2.COLOR_BGR2RGB)) 
    old_im = Image.open(t)
    #old_im = Image.open('C:/Users/Admin/Desktop/img/1.jpg')
    old_size = old_im.size
    print(old_size)
    
    new_size = (250, 250)
    print(new_size)
    new_im = Image.new("RGB", new_size,"black")   ## luckily, this is already black!
    x = int((new_size[0]-old_size[0])/2)
    y = int((new_size[1]-old_size[1])/2)
    new_im.paste(old_im, (x,y))
    
    return new_im




test_dir = "C:/Users/Admin/Desktop/img/nir_fen/"
test_dir1 = "C:/Users/Admin/Desktop/img/rgb_fen/"


save_dir = "C:/Users/Admin/Desktop/img/resize_nir"
save_dir1 = "C:/Users/Admin/Desktop/img/resize_rgb"

img_names = os.listdir(test_dir)

for img_name in img_names:
    t = os.path.join(test_dir, img_name)
    t1 = os.path.join(test_dir1, img_name)
    #img_names = os.listdir(test_dir)
    new_im = jiabaibian(t)
    new_im1 = jiabaibian(t1)
    #image_data = np.asarray(new_im)
    #Image=cv2.cvtColor(image_data,cv2.COLOR_BGR2RGB) 
    
    #cv2.imwrite(save_dir + "/" + '%d.jpg'%a, proposal)
    new_im.save(save_dir + "/" + img_name)
    new_im1.save(save_dir1 + "/" + img_name)
    
    


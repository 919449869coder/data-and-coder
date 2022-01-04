import os
import numpy as np 
import cv2
from matplotlib import pyplot as plt

test_rgb = "C:/Users/Admin/Desktop/img/rgb/"
test_nir = "C:/Users/Admin/Desktop/img/nir/"
#img = cv2.imread("C:/Users/Admin/Desktop/img/222/48.jpg")
save_dir = "C:/Users/Admin/Desktop/img/bad/"

img_names = os.listdir(test_rgb)
for img_name in img_names:
    img_rgb = cv2.imread(os.path.join(test_rgb, img_name))
    img_rgb = img_rgb[:,:,0]
    img_nir = cv2.imread(os.path.join(test_nir, img_name))
    
    ret,mask=cv2.threshold(img_rgb,220,1, cv2.THRESH_BINARY_INV)
    GrayImage_nir=cv2.cvtColor(img_nir,cv2.COLOR_BGR2GRAY) 
    GrayImage_nir = GrayImage_nir[0:964,:]
    last_img = mask*GrayImage_nir
    
    cv2.imwrite(save_dir + "/" + img_name, last_img)

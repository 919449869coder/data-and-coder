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

test_dir_rgb = "C:/Users/Admin/Desktop/img/zengqiang_rgb/"
test_dir_nir = "C:/Users/Admin/Desktop/img/zengqiang_nir/"
save_dir = "C:/Users/Admin/Desktop/img/zengqiang_rgb_nir/"

img_names = os.listdir(test_dir_rgb)

for img_name in img_names:
    img_rgb = cv2.imread(os.path.join(test_dir_rgb, img_name))
    img_nir = cv2.imread(os.path.join(test_dir_nir, img_name))
    
    pnir=img_nir[:,:,1]
    p_rgb_nir = cv2.merge([img_rgb,pnir])
    
    f_out = open('C:/Users/Admin/Desktop/img/1.txt', 'r+')
    a = f_out.read()
    a = int(a) + 1
    print(a)
    f_out.seek(0)
    f_out.truncate()
    f_out.write(str(a))
    f_out.close()

    #i = x_min+x_max+y_max+y_min
    cv2.imwrite(save_dir + "/" +'bad'+ '%d.png'%a, p_rgb_nir)
    
f_out = open('C:/Users/Admin/Desktop/img/1.txt', 'r+')
a = f_out.read()
a = 0
f_out.seek(0)
f_out.truncate()
f_out.write(str(a))
f_out.close()
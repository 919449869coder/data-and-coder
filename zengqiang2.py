# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 09:41:07 2021

@author: Admin
"""
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
# img_dag = ImageDataGenerator(rotation_range=30, 
#                              width_shift_range=0.1,
#                              height_shift_range = 0.1, 
#                              shear_range = 0.0, 
#                              zoom_range = 0.0,
#                              brightness_range=None,
#                              vertical_flip = True,
#                              horizontal_flip = True, 
#                              fill_mode = "constant") #旋
img_dag = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

l=[]
l_nir=[]
img = load_img("C:/Users/Admin/Desktop/img/rgb_fen/1.jpg")  # this is a PIL image
img1 = load_img("C:/Users/Admin/Desktop/img/nir_fen/1.jpg")  # this is a PIL image
y=img_to_array(img1)
x = img_to_array(img) # this is a Numpy array with shape (3, 150, 150)
l.append(x)
l_nir.append(y)
l=np.array(l)
l_nir=np.array(l_nir)
#l = l.reshape((1,) + l.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
 
# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory

f_out = open('C:/Users/Admin/Desktop/img/1.txt', 'r+')
a = f_out.read()
a = int(a) + 10
#print(a)
f_out.seek(0)
f_out.truncate()
f_out.write(str(a))
f_out.close()

image_out_path = "C:/Users/Admin/Desktop/img/zengqiang_rgb/"
image_nir_out_path = "C:/Users/Admin/Desktop/img/zengqiang_nir/"

img_generator = img_dag.flow(l, batch_size=1,shuffle=False,
                                      save_to_dir=image_out_path,
                                      save_prefix = "image", save_format = "png",seed=a)#测试一张图像bath_size=1
img_generator_nir = img_dag.flow(l_nir, batch_size=1,shuffle=False,
                      save_to_dir=image_nir_out_path,
                      save_prefix = "image", save_format = "png",seed=a)#测试一张图像bath_size=1   save_prefix=''名字前缀
        
i = 0
for batch in img_generator:
    i += 1
    if i > 2:
        break  # otherwise the generator would loop indefinitely
i = 0
for batch in img_generator_nir:
    i += 1
    if i > 2:
        break  # otherwise the generator would loop indefinitely
    

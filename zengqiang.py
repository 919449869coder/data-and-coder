import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
import cv2
import os




def date_enhancement(path,path_nir,image_out_path,image_nir_out_path):
    names = os.listdir(path)
    for filename in names:
        a = path+filename
        b = path_nir+filename

        image = load_img(a)
        image_nir = load_img(b)  
        
        image2 = img_to_array(image) #图像转为数组
        image2_nir = img_to_array(image_nir) #图像转为数组
        
        image = np.expand_dims(image2, axis=0) #增加一个维度
        image_nir = np.expand_dims(image2_nir, axis=0) #增加一个维度
        
        img_dag = ImageDataGenerator(rotation_range=30, 
                                     width_shift_range=0.1,
                                     height_shift_range = 0.1, 
                                     shear_range = 0.0, 
                                     zoom_range = 0.0,
                                     brightness_range=None,
                                     vertical_flip = True,
                                     horizontal_flip = True, 
                                     fill_mode = "constant") #旋转，宽度移动范围，高度移动范围，裁剪范围，水平翻转开启，填充模式
        #"constant", "nearest", "reflect" or "wrap"
    
        f_out = open('C:/Users/Admin/Desktop/img/1.txt', 'r+')
        a = f_out.read()
        a = int(a) + 10
        #print(a)
        f_out.seek(0)
        f_out.truncate()
        f_out.write(str(a))
        f_out.close()


        img_generator = img_dag.flow(image, batch_size=1,shuffle=False,
                                      save_to_dir=image_out_path,
                                      save_prefix = "image", save_format = "jpg",seed=a)#测试一张图像bath_size=1
        img_generator_nir = img_dag.flow(image_nir, batch_size=1,shuffle=False,
                              save_to_dir=image_nir_out_path,
                              save_prefix = "image", save_format = "jpg",seed=a)#测试一张图像bath_size=1   save_prefix=''名字前缀
        count =0 #计数器
        for img in img_generator:
            count += 1
            if count == 8:  #生成多少个样本后退出
                break
        countt =0 #计数器
        for img in img_generator_nir:
            countt += 1
            if countt == 8:  #生成多少个样本后退出
                break


if __name__=="__main__":
    path="E:/数据集/玉米粒数据集3/原始图像160增强/yushi/rgb/"
    path_nir="E:/数据集/玉米粒数据集3/原始图像160增强/yushi/nir/"

    #image_path ="C:/Users/Admin/Desktop/jietu/wan.2.jpg"
    image_out_path = "E:/数据集/玉米粒数据集3/原始图像160增强/rgb/"
    image_nir_out_path = "E:/数据集/玉米粒数据集3/原始图像160增强/nir/"
    date_enhancement(path,path_nir,image_out_path,image_nir_out_path)
    
    f_out = open('C:/Users/Admin/Desktop/img/1.txt', 'r+')
    a = f_out.read()
    a = 0
    #print(a)
    f_out.seek(0)
    f_out.truncate()
    f_out.write(str(a))
    f_out.close()


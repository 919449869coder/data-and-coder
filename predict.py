import numpy as np
import cv2
from keras import backend as K
###导入自定义包 加这句话
import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
###
#from modell.VGG16 import VGG16
import os
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import sys,os
# sys.path.append('需要作为模块引入的路径')
# 添加当前路径的前一级文件作为源文件夹
# path = os.path.dirname(os.path.dirname('zhongzijiance/modelss')) 
# print(path)
# sys.path.append(path)


#K.set_image_dim_ordering('tf')
def deeplearn_class(img,model):

    cls_names = ["good", "bad"]
    #image_size = (224,224)
    #model = VGG16(2)
    #model.load_weights('D:/study/study2/traffic/yumi/yumi-detection-master/zhongzijiance/save_model/model_vgg.h5')

    images = []
    labels = []

    res=cv2.resize(img,(224,224),interpolation=cv2.INTER_LINEAR) # 双线性插值
    img = image.img_to_array(res)
    x = np.expand_dims(img, axis=0)
    #x = x/255
    x = preprocess_input(x)
    #第一次用的网络是在model=Sequential()下添加模块的的方法，也就是所谓的顺序模型
    #result = model.predict_classes(x,verbose=0) #predict和predict_classes两个预测函数，前一个是返回的精度，后面的是返回的具体标签
    #编写好网络结构后使用model=Model()综合起来的方法
    result = model.predict(x)

    cls_prop = np.round(result, 2)#保留两位小数
    result1=np.argmax(result,axis=1)
    cls_num = result1[0]
    #最大索引
    cls_numnum = max(cls_prop[0])
    cls_name = cls_names[result1[0]]

    #print(f,cls_name,cls_num)
    print(cls_name,cls_num,)
    return cls_name,cls_num,cls_numnum
    
    
    
    
    
    
    
    
    
    
    
    
    
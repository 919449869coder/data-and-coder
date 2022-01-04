import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras import callbacks
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
from keras.models import Sequential, model_from_yaml, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
from keras.utils import np_utils, plot_model
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import preprocess_input, decode_predictions
import skimage.io
from scipy.io import loadmat
###导入自定义包 加这句话
import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
###
from modell.VGG16 import VGG16,VGG19
from modell.resnet50 import ResNet50
from modell.alexnet import Alexnet,Alexnet2
#from modell.inception3 import InceptionV3
from modell.Inception11 import InceptionV3
from modell.MobileNet import MobileNet
from modell.vgg_restnet import VGG16_ResNet50
from modell.Densenet import DenseNet169
from modell.Densenet import DenseNet121
from modell.restnet_inception import resnet_inception
from modell.Inception_ResnetV2 import InceptionResNetV2
from modell.MobileNetV2 import MobileNetV2
from modell.Xception import Xception
import random
import os
import numpy as np 
import cv2

#https://www.cnblogs.com/softmax/p/7739141.html
np.random.seed(7)
img_h, img_w = 224, 224
image_size = (224, 224)#299#224
nbatch_size = 16
nepochs = 100
nb_classes = 2


def load_data():
    #path = 'C:/Users/Admin/Desktop/mat/train/'
    path = 'D:/study/study2/yumi_nir/data/train/'
    files = os.listdir(path)
    images = []
    labels = []
    for f in files:
        #image_size = (224, 224)
        img_path = path + f
        
        scr = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
        
        res=cv2.resize(scr,(224,224),interpolation=cv2.INTER_LINEAR) # 双线性插值
        
        img_array = image.img_to_array(res)
        
        # img = image.load_img(img_path, target_size=image_size)
        # img_array = image.img_to_array(img)
        
        images.append(img_array)
        ##good_为0
        if 'good' in f:
            labels.append(0)
        else:
            labels.append(1)
    
    data = np.array(images)   #1000*150*150*3
    labels = np.array(labels)  

    labels = np_utils.to_categorical(labels, 2)
    return data, labels

def main():
    # 模型保存的位置
    log_dir = "D:/study/study2/yumi_nir/save_model/"
    print("compile.......")
    #sgd = Adam(lr=1e-3)
    sgd = SGD(lr=1e-3, momentum=0.9, nesterov=True)
    #model = VGG16(nb_classes)
    #model = InceptionV3(input_shape=[224,224,4],classes=2)
    #model = Xception(input_shape=[224,224,4],classes=2)
    #model = Alexnet2(nb_classes)
    #model = ResNet50(input_shape=[224,224,4],classes=nb_classes)
    #model = MobileNet(input_shape=(224, 224, 4))
    #model = MobileNetV2(input_shape=[224,224,4],classes=2)
    #model = VGG16_ResNet50(classes=nb_classes)
    #model = VGG19(nb_classes)
    #model = resnet_inception(classes=2)
    #model = InceptionResNetV2(classes=2)
    model = DenseNet121()
    #model = DenseNet169()
    model.summary()
    #model.load_weights("quanzhong/vgg16_weights_tf_dim_ordering_tf_kernels.h5",by_name=True,skip_mismatch=True)
    model.load_weights("quanzhong/densenet121_weights_tf_dim_ordering_tf_kernels.h5",by_name=True,skip_mismatch=True)
    #model.load_weights('save_model/ep070-loss0.000-val_loss0.263-val_acc0.965.h5')
    #vgg16_weights_tf_dim_ordering_tf_kernels

    # # 指定训练层
     # trainable_layer = 175
    # for i in range(0,len(model.layers)-4):
    #     model.layers[i].trainable = False  

    model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
    #model.compile(optimizer=SGD(lr=1e-4,momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])

    print("load_data......")
    images, labels = load_data()
    #images /= 255
    images = preprocess_input(images)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, shuffle=True)
    print(x_train.shape,y_train.shape)  #800，150*150*3    x为图片  y为标签

    print("train.......")
    tbCallbacks = callbacks.TensorBoard(log_dir='D:/study/study2/yumi_nir/logs',write_graph=True)
    #tensorboard --logdir=D:/study/study2/traffic/yumi/yumi-detection-master/zhongzijiance/logs
     # 保存的方式，3世代保存一次
    checkpoint_period1 = ModelCheckpoint(
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-val_acc{val_acc:.3f}.h5',
                                    monitor='acc', 
                                    save_weights_only=False, 
                                    #save_best_only=True, 
                                    period=10
                                )
    reduce_lr = ReduceLROnPlateau(
                            monitor='val_loss', 
                            factor=0.5, 
                            patience=10, 
                            verbose=1,
                            min_lr=1e-6
                        )
    model.fit(x_train, y_train, batch_size=nbatch_size,epochs=nepochs, verbose=1, validation_data=(x_test, y_test), callbacks=[tbCallbacks,reduce_lr,checkpoint_period1])
    #verbose = 0，在控制台没有任何输出verbose = 1 ：显示进度条verbose =2：为每个epoch输出一行记录 initial_epoch=50  reduce_lr,
    print("evaluate......")
    # scroe, accuracy = model.evaluate(x_test, y_test, batch_size=nbatch_size)
    # print('scroe:', scroe, 'accuracy:', accuracy)

    # yaml_string = model.to_yaml()
    # with open('D:/study/classification/操作/modeljieguo/rgb_nir.yaml', 'w') as outfile:
    #     outfile.write(yaml_string)
    #model.save_weights(log_dir+'alexnet.h5')


if __name__ == '__main__':
    #VGG16(nb_classes)
    main()
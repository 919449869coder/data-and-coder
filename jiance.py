import os
import numpy as np 
import cv2
from skimage import feature as ft 
#from sklearn.externals import joblib
import joblib
from matplotlib import pyplot as plt
import copy
from timeit import default_timer as timer
from skimage.feature import greycomatrix, greycoprops
from sklearn import preprocessing
from skimage import morphology,measure
from PIL import Image 
###导入自定义包 加这句话
import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
###
from modell.VGG16 import VGG16
from modell.alexnet import Alexnet2
from modell.resnet50 import ResNet50
from modell.inception3 import InceptionV3
#from modell.Inception11 import InceptionV3
from predict import deeplearn_class
from modell.MobileNet import MobileNet
from modell.vgg_restnet import VGG16_ResNet50
from modell.Densenet import DenseNet121
from modell.restnet_inception import  resnet_inception
from modell.Xception import Xception




cls_names = ["good", "bad"]
classes_num = {"good":0,"bad":1}

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

def watershed_show_obj(GrayImage,img,color):
  ###改改改
  ret,img_bin=cv2.threshold(GrayImage,color,255, cv2.THRESH_BINARY_INV)
  #ret,thresh1=cv2.threshold(GrayImage,color,255,cv2.THRESH_BINARY)
  #img_bin = ~thresh1
    
  # imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  # hmin = np.array([0, 0,  50 ])
  # hmax = np.array([180, 250,240])
  # img_bin = cv2.inRange(imgHSV,hmin, hmax)

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



#  plt.imshow(img)
# cv2.imshow("ret",img)
# cv2.waitKey(0) 
###形态特征




def object_detection_dir(test_dir, write_dir, result_txt):
    """test the images in test direction.
    Args:
        test_dir: test directory.
        write_dir: write directory of detection results.
        result_txt: a txt file used to save　all the detection results.
    Return: 
        none
    """
    #model = VGG16(2)
    #model = Alexnet2(2)
    #model = MobileNet(input_shape=(224, 224, 4))
    #model = ResNet50(input_shape=[224,224,4],classes=2)
    #model = InceptionV3(input_shape=[224,224,4],classes=2)
    #model = VGG16_ResNet50(classes=2)
    #model = DenseNet121()
    model = Xception(input_shape=[224,224,4],classes=2)
    #model = resnet_inception(classes=2)
    model.load_weights(r'save_model\ep100-loss0.000-val_loss0.178-val_acc0.967')

    start = timer()
    img_names = os.listdir(test_dir)
    img_names = [img_name for img_name in img_names if img_name.split(".")[-1] == "jpg"]
    
    if os.path.exists(result_txt):
        os.remove(result_txt)
    f = open(result_txt, "a")
    for index, img_name in enumerate(img_names):
        if index%50 == 0:
            print ("total test image number = ", len(img_names), "current image number = ", index)
        
        row_data = os.path.join(test_dir, img_name) + " "
        save_path = os.path.join(write_dir,img_name.split(".")[0]+"_result.jpg")
        
        p=os.path.join(test_dir, img_name)
        pp=os.path.join(test_dir_nir, img_name)
        img = cv2.imread(p)
        img1 = img.copy()
        GrayImage = img[:,:,0]
        img_nir = cv2.imread(pp)
        GrayImage_nir=cv2.cvtColor(img_nir,cv2.COLOR_BGR2GRAY) 
        ###图片从这里读进来
        ret1,mask=cv2.threshold(GrayImage,220,1, cv2.THRESH_BINARY_INV)
        GrayImage_nir = GrayImage_nir[0:964,:]
        last_img = mask*GrayImage_nir
        
        last_img1 = cv2.merge([last_img,last_img,last_img])

        markers,ret=watershed_show_obj(GrayImage,img,color=210) #210
        #markers_nir,ret_nir=watershed_show_obj(GrayImage_nir,img_nir,color=60)
        
        height,width=get_img_size(markers)
        area=get_all_class_coordinate(ret,height,width,markers)
        rects,p=obj_corner_coordinate(area)

        img_bbx = img1.copy()
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

            ############
            m=rects.index(rect)+2                         
            oneimg = copy.deepcopy(img1)
            oneimg_nir = copy.deepcopy(last_img1)
            
            oneimg[markers != m] = [0,0,0]      #判断像素点值不是标签值得变为o
            oneimg_nir[markers != m] = [0,0,0]      #判断像素点值不是标签值得变为o
        
            x_min,y_min,x_max,y_max = rectss        
            proposal = oneimg[x_min:x_max,y_min:y_max]
            proposal_nir = oneimg_nir[x_min:x_max,y_min:y_max]
            ########
            proposal = jiabaibian(proposal)
            proposal_nir = jiabaibian(proposal_nir)
            proposal = cv2.cvtColor(np.asarray(proposal),cv2.COLOR_RGB2BGR) 
            proposal_nir = cv2.cvtColor(np.asarray(proposal_nir),cv2.COLOR_RGB2BGR) 
            pnir=proposal_nir[:,:,1]
            p_rgb_nir = cv2.merge([proposal,pnir])
                    

#改          

            #cls_prop = hog_extra_and_svm_class(proposal, clf)
            cls_name,cls_num,cls_numnum = deeplearn_class(p_rgb_nir,model)
            #print(cls_prop)
            # cls_prop = np.round(cls_prop, 2)#保留两位小数
            # cls_num = np.argmax(cls_prop)#返回一个numpy数组中最大值的索引值
            # cls_name = cls_names[cls_num]
            #print(cls_name) 
            

            if cls_name is  "good":
                row_data += str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + str(cls_num) + " "
                cv2.rectangle(img_bbx,(rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0,255,0), 2)
                cv2.putText(img_bbx, cls_name+str(cls_numnum), (rect[0], rect[1]), 1, 2, (0,0,0),2)
                #各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
            else:
                row_data += str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + str(cls_num) + " "
                cv2.rectangle(img_bbx,(rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0,0,255), 2)
                cv2.putText(img_bbx, cls_name+str(cls_numnum), (rect[0], rect[1]), 1, 2, (0,0,0),2)
                #各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
           
        cv2.imwrite(save_path, img_bbx)
        f.write(row_data+"\n")
    end = timer()
    print(end - start)
    f.close() 



if __name__ == "__main__":
    #start = timer()
    test_dir = "D:/study/study2/yumi_nir/test/JPEGImages"
    test_dir_nir = "D:/study/study2/yumi_nir/test/JPEGImages_nir"
    write_dir = "D:/study/study2/yumi_nir/test/test_results"      #最后结果
    write_bin_dir = "data/test_results_bin"  #二值掩膜
    result_txt = "D:/study/study2/yumi_nir/test/test_result.txt"
    object_detection_dir(test_dir, write_dir, result_txt)
    print ("finished.")
    # end = timer()
    # print(end - start)
    
# model_path = "D:/study/study2/traffic/yumi/yumi-detection-master/svm_model.pkl"
# imgg = cv2.imread("C:/Users/Admin/Desktop/363.jpg")
# imgg1 = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)

# img = cv2.imread("C:/Users/Admin/Desktop/img/163.jpg")
# img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# resize=(64,64)
# img = cv2.resize(img1, resize)

import os
import cv2

image_path = 'D:/study/study2/yumi_nir/data/good_rgb_nir/'
save_path = 'D:/study/study2/yumi_nir/data/good/'
for file in os.listdir(image_path):
    name = file.split('_')

    new_image = save_path + 'good' + '_' + name[0]
    scr = cv2.imread(image_path + file,cv2.IMREAD_UNCHANGED)
    cv2.imwrite(new_image, scr)
    
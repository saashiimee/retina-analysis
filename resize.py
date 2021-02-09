import cv2
import os
import glob

imgs = glob.glob('./datasets/vessel_segmentation/test/images/*.png')
for i in imgs:
    img = cv2.imread(i)
    resize_img = cv2.resize(img, (584, 565))
    cv2.imwrite(i, resize_img)

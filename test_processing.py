# -*- coding: utf-8 -*-
"""
Created on Mon Oct 02 13:47:02 2017

@author: Mirab
"""

import feature_extractor as fe
import data_reader as dr
import cv2

import matplotlib.pyplot as plt
import re
import numpy as np
import scipy.ndimage.morphology
import copy

import skimage.io

   
ext = fe.HOG()

dataset = dr.DATAset()
config = dataset.config
dataset.create_dataset_CT()

i = 15
imgname = ext.dataset.orig_images[i]
maskname = re.sub('orig_images', 'masks', imgname)

# pokud chci ten problemovy tetstovaci
imgname = ext.dataset.test_images[-2]
bb = [140, 230, 80, 170]

img = ext.dataset.load_obj(imgname)

#boxes = dr.load_json(config["annotations_path"])
#bb = boxes[imgname][0]#[73, 173, 13, 111]

x, h, y, w = bb

padding = 10
(x, y) = (max(x-padding, 0), max(y-padding, 0))
frame = img[x:h, y:w]



frame = cv2.resize(frame, (54, 54), interpolation=cv2.INTER_AREA)
skimage.io.imshow(frame)
plt.show()

ns = [3, 5, 7, 9, 11, 13, 17, 21]
ms = [11, 15, 19, 23, 27, 31, 35, 39, 41]

ms = [35]

cls = [0.5, 1, 1.5, 2]
tgs = [2, 4, 8]

for n in cls:
    for m in tgs:
        roi = frame
        roi = cv2.bilateralFilter(frame.astype("uint8"), 9, 35, 35)
        
        clahe = cv2.createCLAHE(clipLimit=n, tileGridSize=(m, m))
        roi = clahe.apply(roi.astype("uint8"))
        
        plt.figure()
        plt.imshow(roi, cmap='gray')
        title = str(n)+"_"+str(m)
        plt.title(title)
        plt.show()

        #dr.save_image(roi, 'extractor_test_results/Bilatelar/'+title+".png")
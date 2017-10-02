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


def test_bilateral(frame, ds, cs, vs):
    
    for d in ds:
        for c in cs:
            for v in vs:
                
                roi = cv2.bilateralFilter(frame.astype("uint8"), d, c, v)
                
                plt.figure(figsize=(10, 10))
                plt.imshow(roi, cmap='gray')
                title = str(d)+"_"+str(c)+"_"+str(v)
                plt.title(title)
                
                plt.savefig('extractor_test_results/Bilatelar/bilateral'+title+".png")
                
                skimage.io.imshow(roi)
                plt.show()
                #dr.save_image(roi, 'extractor_test_results/Bilatelar/bilateral'+title+".png")


def test_clahe(frame, cls, tgs):
    
    for n in cls:
        for m in tgs:
    
            clahe = cv2.createCLAHE(clipLimit=n, tileGridSize=(m, m))
            roi = clahe.apply(frame.astype("uint8"))
            
            plt.figure(figsize=(10, 10))
            plt.imshow(roi, cmap='gray')
            title = str(n)+"_"+str(m)
            plt.title(title)
        
            plt.savefig('extractor_test_results/Bilatelar/clahe'+title+".png")
            
            skimage.io.imshow(roi)
            plt.show()

   
ext = fe.HOG()

dataset = dr.DATAset()
config = dataset.config
dataset.create_dataset_CT()

i = 15
#i = 19
imgname = ext.dataset.orig_images[i]
maskname = re.sub('orig_images', 'masks', imgname)

# pokud chci ten problemovy tetstovaci
#imgname = ext.dataset.test_images[-2]
#bb = [140, 230, 80, 170]

img = ext.dataset.load_obj(imgname)

boxes = dr.load_json(config["annotations_path"])
bb = boxes[imgname][0]#[73, 173, 13, 111]

x, h, y, w = bb

padding = 10
(x, y) = (max(x-padding, 0), max(y-padding, 0))
frame = img[x:h, y:w]


frame = cv2.resize(frame, (54, 54), interpolation=cv2.INTER_AREA)
skimage.io.imshow(frame)
plt.show()

# bilateral
ds = [7, 9, 11, 13]
cs = [5, 15, 25, 35, 45, 55, 75, 105]
vs = [15, 35, 55, 75, 125]
cs = range(27, 43)

#ds = [9]
cs = [35]
#vs = [55]
#ms = [35]

# clahe
cls = [0.5, 1, 1.5, 2]
tgs = [2, 4, 8]


test_bilateral(frame, ds, cs, vs)
#test_clahe(frame, cls, tgs)

#roi = cv2.bilateralFilter(frame.astype("uint8"), 9, 15, 15)
#
#plt.figure()
#plt.imshow(roi, cmap='gray')
#title = "lll"
#plt.title(title)
#plt.show()
#
#dr.save_image(roi, 'extractor_test_results/Bilatelar/bilateral'+title+".png")

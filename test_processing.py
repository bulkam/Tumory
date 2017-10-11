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
                
                plt.savefig('extractor_test_results/Bilatelar/'+folder+'bilateral'+title+".png")
                
                skimage.io.imshow(roi)
                plt.show()
                cv2.imwrite('extractor_test_results/Bilatelar/'+folder+'bilateral'+title+".png", 
                            cv2.resize(roi, (432, 432), interpolation=cv2.INTER_AREA))


def test_clahe(frame, cls, tgs):
    
    for n in cls:
        for m in tgs:
    
            clahe = cv2.createCLAHE(clipLimit=n, tileGridSize=(m, m))
            roi = clahe.apply(frame.astype("uint8"))
            
            plt.figure(figsize=(10, 10))
            plt.imshow(roi, cmap='gray')
            title = str(n)+"_"+str(m)
            plt.title(title)
        
            plt.savefig('extractor_test_results/Bilatelar/'+folder+'clahe'+title+".png")
            
            skimage.io.imshow(roi)
            plt.show()


def test_median(frame, ks, additional_title=""):

    for k in ks:
        roi = cv2.medianBlur(frame.astype("uint8"), k)
        title = str(k) + additional_title
        
#        plt.figure(figsize=(10, 10))
#        plt.imshow(roi, cmap='gray')
#        
#        plt.title(title)
#    
#        plt.savefig('extractor_test_results/Bilatelar/median'+title+".png")
#        skimage.io.imshow(roi)
        #plt.show()
        
        cv2.imwrite('extractor_test_results/Bilatelar/'+folder+'median'+title+".png", 
                    cv2.resize(roi, (432, 432), interpolation=cv2.INTER_AREA))


def test_coloring_and_median(frame, mask_frame, back_ks, ks):
    
    for k in back_ks:
        roi = color_background(frame, mask_frame, kernel_size=k)
        roi = cv2.resize(roi, (54, 54), interpolation=cv2.INTER_AREA)
        test_median(roi, ks, additional_title="_coloring-G"+str(k))
        

def get_back_color(blur, mask_frame):
    result = cv2.boxFilter(blur, 0, (37,37))
    #result = cv2.boxFilter(result, 0, (27,27))
    return result

def color_background(frame, mask_frame, kernel_size=51):
    blur = copy.copy(frame)
    
    """ Zde se prebarvuje okoli """
    # obarveni pozadi
    liver = np.mean(frame[mask_frame>0])
    liver = np.median(frame[mask_frame>0])
    # pripadne vykresleni

    # prebarveni okoli
    blur[mask_frame==0] = liver
    
#    liver = get_back_color(blur, mask_frame)
#    blur[mask_frame==0] = liver[mask_frame==0]
    
    blur = cv2.GaussianBlur(blur,(kernel_size, kernel_size), 0)
    #blur = cv2.medianBlur(blur.astype("uint8"), 29)
    blur[mask_frame>0] = frame[mask_frame>0]
    return blur

   
ext = fe.HOG()

dataset = dr.DATAset()
config = dataset.config
dataset.create_dataset_CT()

i = 15
#i = 67
imgname = ext.dataset.orig_images[i]
maskname = re.sub('orig_images', 'masks', imgname)

# pokud chci ten problemovy tetstovaci
#imgname = ext.dataset.test_images[-2]
#bb = [140, 230, 80, 170]

img = ext.dataset.load_obj(imgname)
mask = dr.load_obj(maskname)

boxes = dr.load_json(config["annotations_path"])
bb = boxes[imgname][0]#[73, 173, 13, 111]

x, h, y, w = bb

padding = 10
(x, y) = (max(x-padding, 0), max(y-padding, 0))
frame = img[x:h, y:w]
mask_frame = mask[x:h, y:w]

# bilateral
ds = [7, 9, 11, 13]
cs = [5, 15, 25, 35, 45, 55, 75, 105]
vs = [15, 35, 55, 75, 125]
cs = range(27, 43)

ds = [9]
cs = [35]
vs = [55]
#ms = [35]

# clahe
cls = [0.5, 1, 1.5, 2]
tgs = [2, 4, 8]

ks = [1, 5, 7, 9, 11, 15]
#ks = [11]

back_ks = [9, 15, 25, 27, 29, 31, 33]

""" Priprava obrazku """
neg = bool(0)
folder = "Positive/"

#frame = color_background(frame, mask_frame)
if neg:
    frame = ext.dataset.load_obj(ext.dataset.negatives[-1])
    mask_frame = np.ones(frame.shape)*2
    folder = "Negative/"

frame = cv2.resize(frame, (54, 54), interpolation=cv2.INTER_AREA)
#skimage.io.imshow(frame)
#plt.show()

""" Testy """

#test_bilateral(frame, ds, cs, vs)
#test_clahe(frame, cls, tgs)
test_median(frame, ks)
#test_coloring_and_median(frame, mask_frame, back_ks, ks)

#roi = cv2.bilateralFilter(frame.astype("uint8"), 9, 15, 15)
#
#plt.figure()
#plt.imshow(roi, cmap='gray')
#title = "lll"
#plt.title(title)
#plt.show()
#
#dr.save_image(roi, 'extractor_test_results/Bilatelar/bilateral'+title+".png")

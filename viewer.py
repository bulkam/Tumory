# -*- coding: utf-8 -*-
"""
Created on Sat May 06 16:05:32 2017

@author: mira
"""

import numpy as np
import scipy
import cv2

from matplotlib import pyplot as plt

import skimage
import skimage.io

import time
import pickle
import copy

#from skimage.draw import polygon_perimeter

def show_frame_in_image(gray, box, lw=3, detection=False, blured=False, sigma=3):
    """ Vykresli bounding box do obrazku """
    
    # nacteni bounding boxu
    (y, h, x, w) = box
    
#    img = np.zeros(gray.shape)
#    img[(gray>50)&(gray<300)] = gray[(gray>50)&(gray<300)]-100
#    img[x:h, y:w] = 1
    
    # nacteni snimku
    img = copy.copy(gray)/255
    # rozmazani obrazu mimo okenka
    if blured:
        img = scipy.ndimage.gaussian_filter(img, sigma=sigma)
        img[y:h, x:w] = copy.copy(gray[y:h, x:w])/255
    
    # barva ramecku podle toho, jestli bylo neoc detekovano
    value = float(not detection)
    # vytvoreni ramecku
    img[y:h, x:x+lw] = value
    img[y:h, w-lw:w] = value
    img[y:y+lw, x:w] = value
    img[h-lw:h, x:w] = value
    
    # vykresleni obrazku s rameckem
    cv2.imshow("Frame", img)
    cv2.waitKey(1)
    # cekani, aby se na to dalo divat, pri detekci vetsi zpomaleni
    time.sleep(0.025+(1-value)/20)
    

def show_frames_in_image(img, results, min_prob=0.5, lw=1):
    """ Vykresli obrazek a do nej prislusne framy """
    plt.figure()
    skimage.io.imshow(img, cmap = "gray")
    
    for result in results:
        
        if result["result"][0] > min_prob:
            
            box = result["bounding_box"]
            x, h, y, w = box

            plt.plot([y, w], [x, x], "r", lw = lw)
            plt.plot([y, y], [x, h], "r", lw = lw)
            plt.plot([w, w], [x, h], "r", lw = lw)
            plt.plot([y, w], [h, h], "r", lw = lw)

    plt.show()


def test_only():
    img = None
    with open("datasets/processed/test_images/180_arterial-GT008.pklz", 'rb') as f:
        img = pickle.load(f)
    rr, cc = polygon_perimeter([10, 100, 10, 100],
                           [20, 20, 50, 50],
                           shape=img.shape, clip=True)
    img[rr, cc] = 1000
    skimage.io.imshow(img, cmap = "gray")


def test_only2():
    img = None
    with open("datasets/processed/test_images/180_arterial-GT008.pklz", 'rb') as f:
        img = pickle.load(f)
        
    plt.figure()
    skimage.io.imshow(img, cmap = "gray")
    plt.plot([50, 50], [0, 100],"r", lw = "3")
    plt.show()


if __name__ =='__main__':
    
    test_only2()
    
    
    
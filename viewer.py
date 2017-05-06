# -*- coding: utf-8 -*-
"""
Created on Sat May 06 16:05:32 2017

@author: mira
"""

import numpy as np
import cv2

from matplotlib import pyplot as plt

import skimage
import skimage.io

import time
import pickle

#from skimage.draw import polygon_perimeter

def show_frame_im_image(gray, box):
    """ Vykresli bounding box do obrazku """
    
    (x, h, y, w) = box
    
    img = np.zeros(gray.shape)
    img[(gray>50)&(gray<300)] = gray[(gray>50)&(gray<300)]-100
    img[x:h, y:w] = 1
    
    cv2.imshow("Frame", img)
    cv2.waitKey(1)

    time.sleep(0.025)
    

def show_frames_in_image(img, results):
    """ Vykresli obrazek a do nej prislusne framy """
    plt.figure()
    skimage.io.imshow(img, cmap = "gray")
    
    for result in results:
        
        if result["result"][0] > 0:
            
            box = result["bounding_box"]
            x, h, y, w = box

            plt.plot([y, w], [x, x], "r", lw = "3")
            plt.plot([y, y], [x, h], "r", lw = "3")
            plt.plot([w, w], [x, h], "r", lw = "3")
            plt.plot([y, w], [h, h], "r", lw = "3")

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
    
    
    
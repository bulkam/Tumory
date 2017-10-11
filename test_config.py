# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:51:56 2017

@author: Mirab
"""

import classifier as clas
import feature_extractor as fe
import data_reader as dr
import file_manager as fm

import re
import os
import cv2
import copy
import time
import datetime as dt

import skimage
from skimage.feature import hog as hogg
from skimage import exposure, data
#from skimage.filters import roberts, sobel, scharr, prewitt
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import PCA as PCA
from sklearn.decomposition import TruncatedSVD as DEC
from sklearn.feature_selection import VarianceThreshold as VT

if __name__ =='__main__':

    # cesta k datum
    pos_path = "datasets/PNG/datasets/frames/positives/"
    neg_path = "datasets/PNG/datasets/frames/negatives/"
    hnm_path = "datasets/PNG/datasets/frames/HNM/"
    
    # nacteni seznamu obrazku
    positives = [pos_path + imgname for imgname in os.listdir(pos_path) if imgname.endswith('.png')]# and not ('AFFINE' in imgname)]
    negatives = [neg_path + imgname for imgname in os.listdir(neg_path) if imgname.endswith('.png')]# and not ('AFFINE' in imgname)]        
    hnms = [hnm_path + imgname for imgname in os.listdir(hnm_path) if imgname.endswith('.png')]# and not ('AFFINE' in imgname)]
    
    # Inicializace
    hog = fe.HOG()
    hog.dataset.create_dataset_CT()
    config = hog.dataset.config
    
    # prebarvovani
    colorings = [None, 25, 29, 33]
    
    # preprocessing
    processing_methods = [(cv2.bilateralFilter, [[9, 35, 35]]),
                          (cv2.medianBlur, [7, 9, 11, 13])]

    
    # HoG
    oris = [16, 20, 12]
    ppcs = [8, 6, 10, 4]
    cpbs = [2, 3, 4]
    
    # redukce vektoru priznaku
    decompositions = [PCA(n_components=10), 
                      PCA(n_components=32),
                      PCA(n_components=128),
                      PCA(n_components=512),
                      VT()]
                      
    # pro vsechny mozne metody processingu
#    for method, params in processing_methods: 
#        for param in params:
#            if "bilateral" in method.__name__:
#                processing_method = method(roi, 
#                                           param[0],
#                                           param[1],
#                                           param[2])
#            elif "median" in method.__name__:
#                processing_method = method(roi, param)
                
    
    for ori in oris:
        hog.orientations = ori
        for ppc in ppcs:
            hog.pixels_per_cell = ppc
            for cpb in cpbs:
                hog.cells_per_block = cpb
                for decomposition in decompositions:
                    features = hog.extract_features(to_save=False, 
                                                    PCA_partially=True)
                
                
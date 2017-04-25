# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 18:24:33 2017

@author: mira
"""

import classifier as clas
import feature_extractor as fe
import data_reader as dr

def test_hogs():
    hog = fe.HOG()
    xpath = hog.config_path
    config = hog.dataset.config
    win = hog.sliding_window_size
    
    data = hog.dataset
    
    #print hog.dataset.load_annotated_images()
    TM = hog.extract_features()
    
    return TM


def test_others():
    sift = fe.ORB()
    xpath = sift.config_path
    config = sift.dataset.config
    win = sift.sliding_window_size
    
    data = sift.dataset
    
    #print hog.dataset.load_annotated_images()
    TM = sift.extract_features()
    
    return TM

TM = test_others()
TM = test_hogs()

#for each in TM.keys():
#    if not each.startswith("neg"):
#        print each, len(TM[each]["feature_vect"]), TM[each]["label"]
#    if each.startswith("neg"):
#        print each, len(TM[each]["feature_vect"]), TM[each]["label"]
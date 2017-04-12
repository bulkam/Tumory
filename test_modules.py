# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 18:24:33 2017

@author: mira
"""

import classifier as clas
import feature_extractor as fe
import data_reader as dr

hog = fe.HOG()
path = hog.config_path
config = hog.dataset.config
win = hog.sliding_window_size

data = hog.dataset

#print hog.dataset.load_annotated_images()
TM = hog.exctract_features()
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 18:54:05 2017

@author: mira
"""

import numpy as np
import cv2

from matplotlib import pyplot as plt

import skimage.io

import os
import copy

from scipy import io


from sklearn.svm import SVC

from imutils import paths
import imutils

import random
import cPickle

import data_reader

#TODO: kompletne dodelat potrevne metody... napriklad predikce, pretrenovani, atd.

class Classifier():
    
    def __init__(self, configpath="configuration/", configname="soccer_ball.json", C=0.01):
        
        self.C = C
        self.config_path = configpath + configname
        self.config = self.precti_json(configpath + configname)
        self.dataset = data_reader.DATAset(configpath, configname)
        
    
    def train(self, data, labels):
        """ Natrenuje klasifikator a ulozi jej do souboru """
        
        print " Trenuje se klasifikator ...",
        classifier = SVC(kernel="linear", C = self.C, probability=True, random_state=42)
        classifier.fit(data, labels)
        print "Hotovo"
        
        # ulozi klasifikator do .cpickle souboru
        print " Uklada se klasifikator do souboru .cpickle ...",
        f = open("SVM.cpickle", "w")
        f.write(cPickle.dumps(classifier))
        f.close()
        print "Hotovo"
        


    def load_test_images(self, path):
        """ Nacte testovaci data a klasifikuje je """
        
        images = list(paths.list_images(path))
        for i, imgname in enumerate(images):
            img = skimage.io.imread(imgname, as_grey=True)
            gray = skimage.color.rgb2gray(img)
            gray = skimage.img_as_ubyte(gray)
    
            roi = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
            hist = skimHOG(roi)
            print len(hist)
            
            classifier = cPickle.loads(open("SVM.cpickle").read())
            print classifier.predict(hist)
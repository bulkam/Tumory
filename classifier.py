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

import random
import cPickle

import data_reader
import feature_extractor as fe

#TODO: kompletne dodelat potrebne metody... napriklad predikce, pretrenovani, atd.

class Classifier():
    
    def __init__(self, configpath="configuration/", configname="soccer_ball.json", extractor=fe.SIFT() , C=0.01):
        
        self.config_path = configpath + configname
        self.dataset = data_reader.DATAset(configpath, configname)
        
        self.extractor = extractor
        self.descriptor_type = self.extractor.descriptor_type
        
        self.config = self.dataset.config
        self.C = self.config["C"]        
        
        self.data = None
        self.labels = None
        
    
    def train(self):
        """ Natrenuje klasifikator a ulozi jej do souboru """
        
        data = self.data
        labels = self.labels
        
        print " Trenuje se klasifikator ...",
        classifier = SVC(kernel="linear", C = self.C, probability=True, random_state=42)
        classifier.fit(data, labels)
        print "Hotovo"
        
        # ulozi klasifikator do .cpickle souboru
        print " Uklada se klasifikator do souboru .cpickle ...",
        f = open(self.config["classifier_path"]+"SVM-"+self.descriptor_type+".cpickle", "w")
        f.write(cPickle.dumps(classifier))
        f.close()
        print "Hotovo"

    
    def create_training_data(self):
        """ Vytvori trenovaci data a labely """
        
        print " Nacitam trenovaci data... ", 
        
        TM = self.dataset.precti_json( self.config["training_data_path"]+self.descriptor_type+"_features.json" )
        
        data = list()
        labels = list()
        
        for value in TM.values():
            
            data.append(value["feature_vect"])
            labels.append(value["label"])
        
        self.data = np.vstack(data)
        self.labels = np.array(labels)

        print "Hotovo"
        

    def load_test_images(self, path):
        """ Nacte testovaci data a klasifikuje je """
        
        images = list(paths.list_images(path))
        for i, imgname in enumerate(images):
            img = skimage.io.imread(imgname, as_grey=True)
            gray = skimage.color.rgb2gray(img)
            gray = skimage.img_as_ubyte(gray)
    
            roi = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
            hist = fe.skimHOG(roi)
            print len(hist)
            
            classifier = cPickle.loads(open("SVM.cpickle").read())
            print classifier.predict(hist)
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 17:01:02 2017

@author: mira
"""


import os
import copy
import numpy as np


import skimage.io

import glob
import json

from imutils import paths
import imutils

import random
import pickle
import cPickle


class DATAset:
    def __init__(self, configpath="configuration/", configname="soccer_ball.json"):
        
        self.config_path = configpath + configname
        self.config = self.precti_json(configpath + configname)

        self.orig_images_path = self.config["images_path"]
        self.annotated_images_path = self.config["annotated_images_path"]
        self.negatives_path = self.config["negatives_path"]
        self.test_images_path = self.config["test_images_path"]
        
        self.annotations_path = self.config["annotations_path"] # file with bounding boxes
        
        self.orig_images = list()
        self.annotated_images = list()
        self.negatives = list()
        self.test_images = list()
        
        self.annotations = dict() # bounding boxes
        
        self.features = list()    # HOG, SIFT, SURF, ... features


    def create_dataset(self):
        """ Vytvori cely dataset """
        unprocessed_images_path = self.config["unprocessed_images_path"]
        unprocessed_negatives_path = self.config["unprocessed_negatives_path"]
        unprocessed_test_images_path = self.config["unprocessed_test_images_path"]
        
        print "Vytvarim obrazky..."
        self.prepare_images(unprocessed_images_path, self.orig_images_path, self.orig_images)
        print "Vytvarim negativni obrazky..."  
        self.prepare_images(unprocessed_negatives_path, self.negatives_path, self.negatives)
        print "Vytvarim testovaci obrazky..." 
        self.prepare_images(unprocessed_test_images_path, self.test_images_path, self.test_images)
        print "Zpracovavam anotace..."
        self.annotations = self.load_annotated_images()
        print "Hotovo"

        
    def precti_json(self, name):
        """ Nacte .json soubor a vrati slovnik """
        filepath = os.path.dirname(os.path.abspath(__file__))+"/"+str(name)
        mydata = {}
        with open(filepath) as d:
            mydata = json.load(d)
            d.close()
        return mydata
     
       
    def zapis_json(self, jsondata,  name):
        """ Ulozi slovnik do .json souboru """
        filepath = os.path.dirname(os.path.abspath(__file__))+"/"+str(name)
        with open(filepath, 'w') as f:
            json.dump(jsondata, f)

    
    def save_obj(self, obj, name):
        """ Ulozi data do .pkl souboru """
        filepath = os.path.dirname(os.path.abspath(__file__))+"/"+str(name)
        with open(filepath, 'wb') as f:
            f.write(cPickle.dumps(obj))
            f.close()


    def load_obj(self, name):
        """ Ulozi data do .pkl souboru """
        filepath = os.path.dirname(os.path.abspath(__file__))+"/"+str(name)
        with open(filepath, 'rb') as f:
            return pickle.load(f)

        
    def upload_config(self, configname, new_config):
        """ Aktualizuje konfiguracni soubor .json
            -> prida nove polozky a aktualizuje stare """
        config = dict()
        try:
            config = self.precti_json(configname)
        except:
            pass
        
        for key in new_config.keys():
            config[key] = new_config[key]
        # zapise json do zase do souboru
        self.zapis_json(config, configname)
    
    
    def prepare_images(self, source_path, target_path, processed_images):
        """ Nacte obrazky a ulozi je ve vhodne forme (sede atd.) 
            -> navic je ulozi take do prislusneho seznamu v teto tride """
        images = list(paths.list_images(source_path))
        for i, imgname in enumerate(images):
            img_orig = skimage.io.imread(images[i], 0)
            gray_orig = skimage.color.rgb2gray(img_orig)
            gray_orig = skimage.img_as_ubyte(gray_orig)
            
            image_target_path = target_path + os.path.basename(imgname)
            
            skimage.io.imsave(image_target_path, gray_orig)
            processed_images.append(image_target_path)

    
    def load_annotated_images(self):
        """ Nacte anotovane obrazky a spocita bounding boxy -> 
            -> ty pote vrati a zaorven ulozi do jsonu """
        if not self.orig_images:
            self.orig_images = list(paths.list_images(self.orig_images_path))
        self.annotated_images = list(paths.list_images(self.annotated_images_path))
        
        boxes = dict()
        for i, imgname in enumerate(self.annotated_images):
            
            img_anot = skimage.io.imread(imgname, as_grey=True)
            gray_anot = skimage.color.rgb2gray(img_anot)
            gray_anot = skimage.img_as_float(gray_anot)
            img_orig = skimage.io.imread(self.orig_images[i], as_grey=True)
            gray_orig = skimage.color.rgb2gray(img_orig)
            gray_orig = skimage.img_as_float(gray_orig)
            
            difference = np.abs(gray_anot - gray_orig)
    
            coords = np.where(difference>0.1)
            (y, h, x, w) = min(coords[0]), max(coords[0]), min(coords[1]), max(coords[1])
            boxes[self.orig_images[i]] = {"x":x, "w":w, "y":y, "h":h}
            boxes[self.orig_images[i]] = (y, h, x, w)
            
        self.zapis_json(boxes, self.annotations_path)
        return boxes
    
    

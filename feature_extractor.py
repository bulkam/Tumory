# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 19:13:19 2017

@author: mira
"""

import numpy as np
import cv2

from matplotlib import pyplot as plt
from skimage.feature import hog

import skimage.io

import os
import copy

from scipy import io
from scipy.cluster.vq import *

from sklearn.feature_extraction.image import extract_patches_2d

import random
import cPickle

import data_reader

class Extractor(object):
    
    def __init__(self, configpath="configuration/", configname="soccer_ball.json"):
        
        self.config_path = configpath + configname
        self.dataset = data_reader.DATAset(configpath, configname)
        
        self.dataset.create_dataset()
        
        self.sliding_window_size = self.dataset.config["sliding_window_size"]
        
        self.features = dict()


    def get_roi(self, img, bb, padding=5, new_size=(64, 64)):
        """ Podle bounding boxu vyrizne z obrazku okenko """
        
        (i, h, j, w) = bb
        (i, j) = (max(i-padding, 0), max(j-padding, 0))
        roi = img[i:h+padding, j:w+padding]
        
        # zmeni velikost regionu a vrati ho
        return cv2.resize(roi, new_size, interpolation = cv2.INTER_AREA)
    
   
    def pyramid_generator(self, img, scale=1.5, min_size=(30, 30)):
        """ Postupne generuje ten samy obrazek s ruznymi rozlisenimy """
        
        # nejdrive vrati obrazek v puvodni velikosti
        yield img
        
        min_h, min_w = minSize
        
        # pote zacne vracet zmensene obrazky
        while True:
            img = scipy.misc.imresize(img, 1.0/scale)        # zmensi se obrazek
            height, width = img.shape[0:2]
            
            # pokud je obrazek uz moc maly, zastavi se proces
            if (height < min_h) or (width < min_h):    
                break
            
            # vrati zmenseny obrazek
            yield img


    def sliding_window_generator(self, img, step, window_size):
        """ Po danych krocich o velikost step_size prostupuje obrazem 
            a vyrezava okenko o velikost window_size """
            
        (height, width) = img.shape[0:2]
        (win_width, win_height) = window_size
        h = 0
        while True:
            w = 0
            while True:
                    yield img[h:h+win_height, w:w+win_width]
                    w += step
                    if w+step >= width:
                            break
            h += step
            if h+step >= height:
                    break

       
    def count_sliding_window_size(self, boxes_path):
        """ Spocita velikost sliding window na zaklade prumerne velikost b.b. """
        
        boxes = self.dataset.precti_json(boxes_path)
        widths = []
        heights = []
        
        for box in boxes.keys():
            # Prida se sirka a vyska do seznamu
            (y, h, x, w) = boxes[box]
            heights.append(h-y)
            widths.append(w-x)
        	
        # ze seznamu se spocita prumer jak pro vysku, tak pro sirku
        avg_height, avg_width = np.mean(heights), np.mean(widths)
        print " - prumerna sirka: ", avg_width
        print " - prumerna vyska: ", avg_height
        print " - pomer stran: ", avg_width / avg_height
        width = np.round((avg_width/8))*4
        height = np.round((avg_height/8))*4
            
        return (height, width)


class HOG(Extractor):
    
    def __init__(self, configpath="configuration/", configname="soccer_ball.json", orientations=12, pixelsPerCell=(4, 4), cellsPerBlock=(2, 2)):
        super(HOG, self).__init__(configpath, configname)
        self.orientations = orientations
        self.pixels_per_cell = pixelsPerCell
        self.cells_per_block = cellsPerBlock
    
    
    def skimHOG(self, gray):
        """ Vrati vektor HOG priznaku """
        
        hist = hog(gray, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell,
                            cells_per_block=self.cells_per_block)  # hog je 1 dlouhy vektor priznaku, nesmi tam byt to visualize
        hist[hist<0] = 0
        return hist


    def exctract_features(self):
        """ Spocte vektory HOG priznaku pro trenovaci data a pro negatives ->
            -> pote je olabeluje 1/-1 a ulozi jako slovnik do .json souboru """
            
        features = self.features
        
        print "Nacitaji se Trenovaci data ...",
        
        # Trenovaci data - obsahujici objekty
        for imgname in self.dataset.orig_images:
            
            if self.dataset.annotations.has_key(imgname):
                
                img = skimage.io.imread(imgname)        # nccte obrazek
                box = self.dataset.annotations[imgname] # nacte bounding box
                
                roi = self.get_roi(img, box)            # vytahne region z obrazu
                rois = [roi]
                
                # smycka, kdybych chtel ulozit roi v ruznych natocenich napriklad
                for roi in rois:
                    # extrahuje vektory priznaku regionu
                    features_vect = self.skimHOG(roi)
                    
                    # ulozi se do datasetu
                    features[imgname] = dict()
                    features[imgname]["label"] = 1
                    features[imgname]["feature_vect"] = list(features_vect)
                    
        print "Hotovo"
        print "Nacitaji se Negativni data ...",
        
        # Negativni data - neobsahujici objekty
        negatives = self.dataset.negatives
        for i in xrange(self.dataset.config["number_of_negatives"]):
            # nahodne vybere nejake negativni snimky
            img = cv2.imread(random.choice(negatives))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            patches = extract_patches_2d(gray, tuple(self.sliding_window_size), max_patches = self.dataset.config["number_of_negatives"])
            
            for patch in patches:
                # extrakce vektoru priznaku
                features_vect = self.skimHOG(patch)
                
                # ulozeni do trenovaci mnoziny
                features[imgname] = dict()
                features[imgname]["label"] = -1
                features[imgname]["feature_vect"] = list(features_vect)
                
        print "Hotovo"
        print "Probiha zapis trenovacich dat do souboru", 
        print self.dataset.config["training_data_path"]+"hog_features.json ...",

        # trenovaci data se zapisou se do jsonu
        self.dataset.zapis_json(features, self.dataset.config["training_data_path"]+"hog_features.json")
        
        print "Hotovo"
        
        return features

#TODO komplet extrakce i u ostatnich
class Others(Extractor):
    """ SIFT, SURF, ORB """
    
    def __init__(self, configpath = "configuration/", configname = "soccer_ball.json"):
        
        super(Others, self).__init__(configpath, configname)
        
        self.descriptor_type = str()
        
        
    def extract_features(self, images, negatives):
        """ vytvoreni detektoru a extraktoru """
        feature_detector = cv2.FeatureDetector_create(self.descriptor_type)
        extractor = cv2.DescriptorExtractor_create(self.descriptor_type)
        
        descriptor_list = list()
        
        for imgname in images:
            img = tp.nacti_obrazek(imgname)
            img = tp.BGRtoRGB(img)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
            keypoints = feature_detector.detect(gray)
            keypoints, descriptor = extractor.compute(gray, keypoints)
            descriptor_list.append((imgname, descriptor)) # descriptory maji stejne delky, ale je jich ruzny pocet matice N x 158 napr.    
        
        # prvni deskriptor, pak uz jen pripina dalsi
        descriptors = descriptor_list[0][1]
        for image_path, descriptor in descriptor_list[1:]:
            descriptors = np.vstack((descriptors, descriptor)) 
        
        # provede k.means shlukovani
        k = 100
        voc, variance = kmeans(descriptors, k, 1)
        
        # spocita se histogram priznaku
        im_features = np.zeros((len(imgnames), k), "float32")
        for i in xrange(len(imgnames)):
            words, distance = vq(descriptor_list[i][1],voc)
            for word in words:
                im_features[i][word] += 1
                
        self.features = im_features     
        
        return im_features


class SIFT(Others):
    
    def __init__(self, configpath = "configuration/", configname = "soccer_ball.json"):
        
        super(SIFT, self).__init__(configpath, configname)
        
        self.config_path = configpath + configname
        self.config = self.dataset.precti_json(configpath + configname)
        
        self.descriptor_type = 'SIFT'


class SURF(Others):
    
    def __init__(self, configpath = "configuration/", configname = "soccer_ball.json"):
        
        super(SURF, self).__init__(configpath, configname)
        
        self.config_path = configpath + configname
        self.config = self.dataset.precti_json(configpath + configname)
        
        self.descriptor_type = 'SURF'


class ORB(Others):
    
    def __init__(self, configpath = "configuration/", configname = "soccer_ball.json"):
        
        super(ORB, self).__init__(configpath, configname)
        
        self.config_path = configpath + configname
        self.config = self.dataset.precti_json(configpath + configname)
        
        self.descriptor_type = 'ORB'




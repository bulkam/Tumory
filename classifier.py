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

import viewer
import data_reader
import feature_extractor as fe

#TODO: kompletne dodelat potrebne metody... napriklad pretrenovani, atd.

class Classifier():
    
    def __init__(self, configpath="configuration/", configname="CT.json", extractor=fe.HOG() , C=0.01):
        
        self.config_path = configpath + configname
        
        self.dataset = extractor.dataset#data_reader.DATAset(configpath, configname)
        
        self.extractor = extractor
        self.descriptor_type = self.extractor.descriptor_type
        
        self.config = self.dataset.config
        self.C = self.config["C"]        
        
        self.data = None
        self.labels = None
        
        self.test_classifier = None
        self.test_results = dict()
        
    
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
        
        TM = self.dataset.precti_json(self.config["training_data_path"]+self.descriptor_type+"_features.json" )
        
        data = list()
        labels = list()
        
        for value in TM.values():
            
            data.append(value["feature_vect"])
            labels.append(value["label"])
        
        self.data = np.vstack(data)
        self.labels = np.array(labels)

        print "Hotovo"
    
    
    def classify_frame(self, gray, imgname):
        """ Pro dany obraz extrahuje vektor priznaku a klasifikuje jej """
        
        # extrakce vektoru priznaku
        roi = cv2.resize(gray, tuple(self.extractor.sliding_window_size), interpolation=cv2.INTER_AREA)
        feature_vect = self.extractor.extract_single_feature_vect(roi)
        
        # klasifikace pomoci testovaneho klasifikatoru
        result = list([np.array([self.test_classifier.predict_proba(feature_vect)[0, 1]])])    # klasifikace obrazu
        #result = list([self.test_classifier.predict(feature_vect)])    # klasifikace obrazu
        
        return result

# TODO: popis metody  
# TODO: pevne parametry do configu a menit je tam, tady je jen nacitat
#       -> napriklad vizualizace
    def classify_image(self, gray, imgname, visualization=False):
        """ Pro dany obraz provede: 
        :param: gray: vstupni obrazek, ktery chceme klasifikovat 
        """
        
        # ve vysledcich se zalozi polozka s timto obrazkem a tam budu pridavat vysledky pro jednotlive framy
        self.test_results[imgname] = list()
        # abych mel prehled kolik framu to detekuje
        n_detected = 0
        # nacteni window_size z konfigurace
        window_size = self.config["sliding_window_size"]
        pyramid_scale = self.config["pyramid_scale"]
        min_prob = self.config["min_prob"]
        
        for scaled in self.extractor.pyramid_generator(gray, scale=pyramid_scale):
            
            # spocteni meritka
            scale = float(gray.shape[0])/scaled.shape[0]
            
            for bounding_box, frame in self.extractor.sliding_window_generator(img = scaled, 
                                                                               step = self.config["sliding_window_step"], 
                                                                               window_size = window_size,
                                                                               image_processing=bool(self.config["image_processing"])):
                                                                                   
                # Pokud se tam sliding window uz nevejde, prejdeme na dalsi                
                if frame.shape != tuple(window_size):
                    continue
                
                # klasifikace obrazu
                result = self.classify_frame(frame, imgname)
                
                # spocteni bounding boxu v puvodnim obrazku beze zmeny meritka
                real_bounding_box = (x, h, y, w) = list( ( scale * np.array(bounding_box) ).astype(int) )   
                
                # ulozeni vysledku
                image_result = {"scale": scale,
                                 "bounding_box": real_bounding_box,
                                 "result": list(result[0])}
                                 
                self.test_results[imgname].append(image_result)
                
                # upozorneni na pozitivni data
                if result[0] > min_prob:
                    print "[RESULT] Nalezen artefakt: ", image_result
                    n_detected += 1
                
                # pripadna vizualizace projizdeni slidong window
                if visualization:
                    viewer.show_frame_in_image(gray, real_bounding_box, 
                                               detection=result[0]>min_prob, 
                                               blured=True, sigma=5)
        
        # ulozeni do souboru vysledku
        self.dataset.zapis_json(self.test_results, self.config["test_results_path"])
        
        # pripadna vizualizace
        if visualization:
            viewer.show_frames_in_image(copy.copy(gray), self.test_results[imgname], min_prob=min_prob)
        
        print "[RESULT] Celkem nalezeno ", n_detected, " artefaktu."


    def classify_test_images(self, visualization=False):
        """ Nacte testovaci data a klasifikuje je """
        
        # nacteni testovaneho klasifikatoru
        self.test_classifier = cPickle.loads( open( self.config["classifier_path"]+"SVM-"+self.descriptor_type+".cpickle" ).read() )
        
        imgnames = self.dataset.test_images
        
        for i, imgname in enumerate(imgnames[0:1]):
            
            print "Testovani obrazku ",imgname,"..."
            # nacteni obrazu
            gray = self.dataset.load_image(imgname)
            
            # klasifikace obrazu
            self.classify_image(gray, imgname, visualization)









    

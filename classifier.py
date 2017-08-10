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
import re

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
        
        print "[INFO] Trenuje se klasifikator... ",
        classifier = SVC(kernel="linear", C = self.C, probability=True, random_state=42)
        classifier.fit(data, labels)
        print "Hotovo"
        
        # ulozi klasifikator do .cpickle souboru
        print "[INFO] Uklada se klasifikator do souboru .cpickle ...",
        f = open(self.config["classifier_path"]+"SVM-"+self.descriptor_type+".cpickle", "w")
        f.write(cPickle.dumps(classifier))
        f.close()
        print "Hotovo"

    
    def create_training_data(self):
        """ Vytvori trenovaci data a labely """
        
        print "[INFO] Nacitam trenovaci data... ", 
        
        TM = self.dataset.precti_json(self.config["training_data_path"]+self.descriptor_type+"_features.json" )
        
        data = list()
        labels = list()
        
        for value in TM.values():
            
            data.append(value["feature_vect"])
            labels.append(value["label"])
        
        self.data = np.vstack(data)
        self.labels = np.array(labels)

        print "Hotovo"
    
    
    def store_false_positives(self, features):
        """ Ulozi feature vektory false positivu do trenovaci mnoziny """
        
        print "[INFO] Ukladani false positives mezi trenovaci data... ",
        TM = self.dataset.precti_json(self.config["training_data_path"]+self.descriptor_type+"_features.json")
        # pridavani false positives vektoru
        TM.update(features)
        # ulozeni trenovacich dat
        self.dataset.zapis_json(TM, self.config["training_data_path"]+self.descriptor_type+"_features.json")
        # aktualizace trenovaci mnoziny
        self.create_training_data()
        
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


    def classify_image(self, gray, mask, imgname, visualization=False, HNM=False):
        """ Pro dany obraz provede: 
        Projede vstupni obrazek pomoci sliding window a zmen mmeritka.
        Kazdy frame klasifikuje a ulozi vysledek.
        V pripade HNM zjisti u kazdeho framu, zda se jedna o false positive
            a pokud ano ulozi jej jak do obrazku, tak i do trenovacich dat
            (features).
        Nakonec se vysledky ulozi do .json souboru.
        
        Arguments:
            gray -- vstupni obrazek, ktery chceme klasifikovat
            mask -- maska pro vstupni obrazek
            imgname -- jmeno obrazku
            visualization -- zda ma byt zobrazeny prubeh testovani na obrazku
            HNM -- zda jde o Hard negative mining
        """
        
        # ve vysledcich se zalozi polozka s timto obrazkem a tam budu pridavat vysledky pro jednotlive framy
        self.test_results[imgname] = list()
        # false positives
        false_positives = dict()
        # abych mel prehled kolik framu to detekuje
        n_detected = 0
        # skutecne klasifikovane
        n_positive_bounding_boxes = 0
        
        # nacteni window_size a dalsich parametru z konfigurace
        window_size = self.config["sliding_window_size"]
        pyramid_scale = self.config["pyramid_scale"]
        sliding_window_step = self.config["sliding_window_step"]
        image_preprocessing = bool(self.config["image_preprocessing"])
        
        # minimalni pravdepodobnost framu pro detekci
        min_prob = self.config["min_prob"]
        # minimalni nutne zastoupeni jater ve framu
        min_liver_coverage = self.config["min_liver_coverage"]
        # minimalni nutne zastoupeni artefaktu ve framu - pro HNM
        min_HNM_coverage = self.config["min_HNM_coverage"]
        
        for scaled in self.extractor.pyramid_generator(gray, scale=pyramid_scale):
            
            # spocteni meritka
            scale = float(gray.shape[0])/scaled.shape[0]
            
            for bounding_box, frame in self.extractor.sliding_window_generator(img = scaled, 
                                                                               step = sliding_window_step,
                                                                               window_size = window_size,
                                                                               image_processing=image_preprocessing):
                                                                                   
                # Pokud se tam sliding window uz nevejde, prejdeme na dalsi                
                if frame.shape != tuple(window_size):
                    continue
                
                # klasifikace obrazu
                result = self.classify_frame(frame, imgname)
                
                # spocteni bounding boxu v puvodnim obrazku beze zmeny meritka
                real_bounding_box = (x, h, y, w) = list( ( scale * np.array(bounding_box) ).astype(int) )   
                # zjisteni, zda se ot nachczi v jatrech
                mask_frame = fe.get_mask_frame(mask, real_bounding_box)
                frame_liver_coverage = fe.liver_coverage(mask_frame)
                
                # ulozeni vysledku
                image_result = {"scale": scale,
                                 "bounding_box": real_bounding_box,
                                 "result": list(result[0]),
                                 "liver_coverage": frame_liver_coverage}
                self.test_results[imgname].append(image_result)
                
                # upozorneni na pozitivni data
                if result[0] > min_prob:
                    print "[RESULT] Nalezen artefakt: ", image_result, frame_liver_coverage
                    n_detected += 1
                
                # podminka detekce
                detection_condition = (result[0] > min_prob) and (frame_liver_coverage >= min_liver_coverage)
                # oznaceni jako pozitivni nebo negativni
                self.test_results[imgname][-1]["mark"] = int(detection_condition)
                # pocitani pozitivnich bounding boxu
                n_positive_bounding_boxes += self.test_results[imgname][-1]["mark"]
                
                # pripadne Hard Negative Mining
                if detection_condition and HNM:
                    # zjisteni skutecneho vyskytu atrefaktu
                    frame_artefact_coverage = fe.artefact_coverage(mask_frame)
                    print "Artefact coverage: ", frame_artefact_coverage
                    # pokud je detekovan, ale nemel by byt
                    if frame_artefact_coverage <= min_HNM_coverage:
                        
                        img_id = "false_positive_"+imgname+"_scale="+str(scale)+"_bb="+str(x)+"-"+str(h)+"-"+str(y)+"-"+str(w)
                        print "[RESULT] False positive !!!"
                        # extrakce vektoru priznaku
                        result_roi = cv2.resize(frame, tuple(self.extractor.sliding_window_size), interpolation=cv2.INTER_AREA)
                        result_feature_vect = list(self.extractor.extract_single_feature_vect(result_roi)[0])
                        # ulozeni do false positives
                        false_positives[img_id] = {"feature_vect":result_feature_vect, "label":-1}
                        # ulozeni mezi fp obrazky
                        self.dataset.save_image(frame, self.config["false_positives_path"]+img_id+".png")
                        self.dataset.save_image(frame, self.config["false_positives_path"]+img_id+".pklz")
                        # ulozeni mezi negatives
                        if bool(self.config["FP_to_negatives"]):
                            self.dataset.save_image(frame, self.config["negatives_path"]+img_id+".pklz")
                            
                            
                # pripadna vizualizace projizdeni slidong window
                if visualization:
                    viewer.show_frame_in_image(gray, real_bounding_box, 
                                               detection=detection_condition, 
                                               blured=True, sigma=5, 
                                               mask=mask)
        
        # ulozeni do souboru vysledku
        self.dataset.zapis_json(self.test_results, self.config["test_results_path"])
        
        # non-maxima suppression
        detected_boxes = self.non_maxima_suppression(imgname)
        
        # pripadna vizualizace
        if visualization:
            viewer.show_frames_in_image(copy.copy(gray), self.test_results[imgname], 
                                        min_prob=min_prob, min_liver_coverage=min_liver_coverage)
            viewer.show_frames_in_image_nms(copy.copy(gray), detected_boxes)
        
        if HNM:
            self.store_false_positives(false_positives)
            print "[RESULT] Celkem nalezeno ", len(false_positives), "false positives."

        
        print "[RESULT] Celkem nalezeno ", n_detected, " artefaktu."
        print "[RESULT] ", n_positive_bounding_boxes, " bounding boxu nakonec vyhodnoceno jako pozitivni."


    def classify_test_images(self, visualization=False):
        """ Nacte testovaci data a klasifikuje je """
        
        # nacteni testovaneho klasifikatoru
        self.test_classifier = cPickle.loads( open( self.config["classifier_path"]+"SVM-"+self.descriptor_type+".cpickle" ).read() )
        
        imgnames = self.dataset.test_images
        
        for i, imgname in enumerate(imgnames[1:2]):
            
            print "[INFO] Testovani obrazku ", imgname, "..."
            # nacteni obrazu
            gray = self.dataset.load_image(imgname)
            # nacteni masky
            maskname = re.sub("test_images", "masks", imgname)
            mask = self.dataset.load_image(maskname)
            
            # klasifikace obrazu
            self.classify_image(gray, mask, imgname, visualization=visualization)
    
    # TODO:
    def hard_negative_mining(self, visualization=False):
        """ Znovu projede tranovaci data a false positives ulozi do negatives """
        
        # nacteni testovaneho klasifikatoru
        self.test_classifier = cPickle.loads( open( self.config["classifier_path"]+"SVM-"+self.descriptor_type+".cpickle" ).read() )
        
        imgnames = self.dataset.orig_images
        
        for i, imgname in enumerate(imgnames[11:11]):
            
            print "[INFO] Testovani obrazku ", imgname, "..."
            # nacteni obrazu
            gray = self.dataset.load_image(imgname)
            # nacteni masky
            maskname = re.sub("orig_images", "masks", imgname)
            mask = self.dataset.load_image(maskname)
            
            # klasifikace obrazu
            self.classify_image(gray, mask, imgname, HNM=True, visualization=visualization)
        
        # ted na negativech
        imgnames = self.dataset.HNM_images
        print imgnames
        
        for i, imgname in enumerate(imgnames[40:41]):
            
            print "[INFO] Testovani obrazku ", imgname, "..."
            # nacteni obrazu
            gray = self.dataset.load_image(imgname)
            # nacteni masky
            maskname = re.sub("hard_negative_mining", "masks", imgname)
            mask = self.dataset.load_image(maskname)
            
            # klasifikace obrazu
            self.classify_image(gray, mask, imgname, HNM=True, visualization=visualization)
    
    
    def create_boxes_nms(self,imgname):
        """ Vybere pravi bounding boxy pro dany obrazek pro nms """
        
        results = self.dataset.precti_json(self.dataset.config["test_results_path"])
        # inicializace boxu a jejich pravdepodobnosti
        boxes = list()
        probs = list()
        
        for result in results[imgname]:
            # ukladani jen tech pozitivnich
            if result["mark"] == 1:
                boxes.append(np.hstack(result["bounding_box"]))
                probs.append(result["result"][0])
        
        boxes = np.vstack(boxes).astype(float)
        probs = np.hstack(probs)
        
        return boxes, probs
    
    # TODO: zkouset optimalni prah
    def non_maxima_suppression(self, imgname, overlap_thr=0.01):
        """ Provede redukci prekryvajicich se bounding boxu """
        
        overlap_thr = self.dataset.config["NMS_overlap_thr"]
        # nacteni pozitivnich bounding boxu a jejich ppsti
        boxes, probs = self.create_boxes_nms(imgname)
        # nacteni jednotlivych souradnic bounding boxu
        ys, hs, xs, ws = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        bb_area = (hs-ys+1)*(ws-xs+1)
        # serazeni indexu podle pravdepodobnosti
        indexes = np.argsort(probs)
        new_indexes = list()
        
        while True:
            # vytazeni indexu bounding boxu s nejvyssi ppsti
            i = indexes[-1]
            new_indexes.append(i)
            index = indexes[:-1]
            
            ys2 = np.maximum(ys[i], ys[index])
            hs2 = np.minimum(hs[i], hs[index])
            xs2 = np.maximum(xs[i], xs[index])
            ws2 = np.minimum(ws[i], ws[index])
            
            new_area = np.maximum(0, ws2-xs2+1) * np.maximum(0, hs2-ys2+1)
            # vypocet prekryti
            overlap = new_area.astype(float) / bb_area[index]
            # odstraneni indexu boxu s vysokym prekrytim
            indexes_to_delete = np.where(overlap > overlap_thr)[0]
            indexes = np.delete(index, indexes_to_delete)
            
            # zastavit po vyprazdneni
            if len(indexes) == 0:
                break

        print "[RESULT] Pocet pozitivnich bounding boxu zredukovan na ", len(new_indexes)
        #print new_indexes
        # vrati vybrane bounding boxy
        return boxes[new_indexes].astype(int)
            
        

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:51:56 2017

@author: Mirab
"""

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


class Tester():
    
    def __init__(self, configpath="configuration/", configname="CT.json"):
        
        # cesta k datum
        self.pos_path = "datasets/PNG/datasets/frames/positives/"
        self.neg_path = "datasets/PNG/datasets/frames/negatives/"
        self.hnm_path = "datasets/PNG/datasets/frames/HNM/"
        
        # Inicializace
        self.manager = fm.Manager()
        self.extractor = fe.HOG()
        self.dataset = self.extractor.dataset
        #self.dataset.create_dataset_CT()
        self.config = self.dataset.config
        
        self.parentname = "extractor_test_results/All/"
        self.childname = ""
        
        
    def create_paths(self, parentpath="extractor_test_results/All/"):
        """ Vytvori vsechny potrebne slozky a nektere soubory do nich ulozi """
    
        parentpath = self.parentname    
        
        paths_to_create = ["evaluation", "data_generator", "scripts",
                           "scripts/boxes", "scripts/config"]
                           
        for path in paths_to_create:
            self.manager.make_folder(parentpath + path)
        
        # zkopirovani skriptu
        fm.copyfile("test_config.py", parentpath+"test_config.py")
        # konfigurace a bounding_boxy
        fm.copyfile(self.config["annotations_path"],
                    parentpath+"scripts/boxes/boxes.json")
        fm.copyfile(self.dataset.config_path,
                    parentpath+"scripts/config/CT.json")
        # ulozeni vsech skriptu
        scripts = [name for name in fm.os.listdir(".") if name.endswith('.py')]
        for script in scripts:
            fm.copyfile(script, parentpath+"scripts/"+script)
    
    
    def backup_test_results(self, positives, negatives, 
                            targetname="extractor_test_results/"):
        """ Zalohuje vsechny vysledky testu """
        
        print "Zalohuji vysledky..."
        
        t = time.time()
        tstamp = str(dt.datetime.fromtimestamp(t))
        tstamp = re.sub(r'\s', '__', tstamp)
        tstamp = re.sub(r'[\:\.]', '-', tstamp)
    
        destination = targetname + tstamp + "/"
        
        # zapis nazvu obrazku
        fnames = {"positives": positives,
                  "negatives": negatives}
        dr.zapis_json(fnames, self.parentname+"image_names.json")
        
        # zkopirovani cele cesty
        fm.copytree(targetname+"All", destination)
            
        print "Hotovo"
    
    
    def get_new_classifier(self):
        """ Vytvori a vrati instanci SVC """
        
        return SVC(kernel="linear", C = 0.15, probability=True, random_state=42)
    
    
    def get_methodname(self, method):
        """ Vrati stringovy label, ktery popisuje metodu """
        
        name = str(method)
        name = re.sub("[\(\)\<\>]", "#", name)
        name = re.sub("[\,\s,\:,\.\'\"]", "-", name)
        
        return name
         
         
    def cross_validation(self, X, y, cv_scorings=None, cv=7):
        """ Provede cross-validaci pro dana data """
        
        #self.dataset.log_info("[INFO] Cross validation...")
        print "[INFO] Cross validation..."
        
        # pokud nejsou definovane scorings, tak je nacist z configu
        if cv_scorings is None:
            cv_scorings = self.config["cv_scorings"]
        
        print "Celkem dat: "
        print "   " + str( len([s for s in y if s > 0]) ) + " pozitivnich"
        print "   " + str( len([s for s in y if s < 0]) ) + " negativnich"
        
        # pro moc velka data zmensit pocet provedeni cross_validace
        if X.shape[1] > 100:
            cv = 5
            if X.shape[1] > 250:
                cv = 3
        
        # vypocet skore -> hodnoty test_ odpovidaji hodnotam cross_val_score
        scores = cross_validate(self.get_new_classifier(),  # vytvori novy klasifikator
                                X, y,
                                scoring=cv_scorings,
                                cv=cv)
                                
        for key in scores.keys():
            scores[key] = list(scores[key])
        
        # vypsani vysledku
        print "[RESULT] Vysledne skore: "
        for key, value in scores.items():
            if "test" in key or "time" in key:
                print "    - ", key, ":", np.mean(value)
            
        # ulozeni vysledku ohodnoceni
        dr.zapis_json(scores, self.parentname+"/evaluation/CV_"+self.childname+".json")
        
        # zalogovani zpravy o ukonceni
        #self.dataset.log_info("      ... Hotovo.")
        
        # confussion matrix
        
    
    def fit_methods(self, positives, negatives, decompositions, n_for_fit=2000):
        
        print "Extrahuji data pro redukci dimenzionality... ",
                        
        P = len(positives)
        N = len(negatives)    
        
        each_img = max( (P + N) // (n_for_fit * 2), 1) 
        
        Xr = list()
        yr = list()
        
        for i, imgname in enumerate(positives):
            if i % each_img == 0:
                img = dr.load_image(imgname)
                feature_vect = self.extractor.extract_single_feature_vect(img)
                Xr.append(feature_vect)
                yr.append(1)
        
        for i, imgname in enumerate(negatives):
            if i % each_img == 0:
                img = dr.load_image(imgname)
                feature_vect = self.extractor.extract_single_feature_vect(img)
                Xr.append(feature_vect)
                yr.append(-1)
        
        Xr = np.vstack(Xr)
        yr = np.array(yr)
        
        for dec in decompositions:
            dec.fit(Xr, yr)
        
        print "Hotovo", 
        print "Data shape: ", Xr.shape
        print "Celkem dat: "
        print "   " + str( len([s for s in yr if s > 0]) ) + " pozitivnich"
        print "   " + str( len([s for s in yr if s < 0]) ) + " negativnich"
        
        return decompositions
    
    
    def extract_data(self, positives, negatives):
        
        print "Extrahuji data..."
    
        X = list()
        y = list()
                    
        for i, imgname in enumerate(positives):
            img = dr.load_image(imgname)
            feature_vect = self.extractor.extract_single_feature_vect(img)
            X.append(feature_vect)
            y.append(1)
    
        for i, imgname in enumerate(negatives):
            img = dr.load_image(imgname)
            feature_vect = self.extractor.extract_single_feature_vect(img)
            X.append(feature_vect)
            y.append(-1)
        
        X = np.vstack(X)
        y = np.array(y)
        
        print "Hotovo", 
        print "Data shape: ", X.shape
        print "Celkem dat: "
        print "   " + str( len([s for s in y if s > 0]) ) + " pozitivnich"
        print "   " + str( len([s for s in y if s < 0]) ) + " negativnich"
            
        return X, y


if __name__ =='__main__':
    
    t = time.time()
    
    # inicializace
    tester = Tester()
    hog = tester.extractor
    
    # vytvoreni potrebnych cest
    tester.create_paths()
    
    # nacteni seznamu obrazku
    positives = [tester.pos_path + imgname for imgname in os.listdir(tester.pos_path) if imgname.endswith('.png')]#  and not ('AFFINE' in imgname)]
    negatives = [tester.neg_path + imgname for imgname in os.listdir(tester.neg_path) if imgname.endswith('.png')]#  and not ('AFFINE' in imgname)]        
    #hnms = [tester.hnm_path + imgname for imgname in os.listdir(tester.hnm_path) if imgname.endswith('.png')]# and not ('AFFINE' in imgname)]
    
    """ Nastaveni parametru """
    # HoG
    oris = [12, 16, 20]
    ppcs = [10, 8, 6, 4]
    cpbs = [2, 3, 4]
    
#    oris = [12, 16]
#    ppcs = [10, 8, 6]
#    cpbs = [2, 3]
    
#    oris = [12]
#    ppcs = [10]
#    cpbs = [2]
    
    # musi byt presne napasovane na seznam decompositions !!!
    dec_fvls = [10, 32, 128, 512]# nastavt na nulu, pokud nenastavujeme pocet features
    # redukce vektoru priznaku
    decompositions = [PCA(n_components=10), 
                      PCA(n_components=32),
                      PCA(n_components=128),
                      PCA(n_components=64)]
    
    """ Proces testovani vsech parametru """
    max_iters = len(oris) * len(ppcs) * len(cpbs) * len(decompositions)
    iters = 0
    
    for ori in oris:
        for ppc in ppcs:
            for cpb in cpbs:
                
                print "Testuje se: ", ori, "-", ppc, "-", cpb
                
                # nastaveni kongigurace feature extractoru (HoGu)
                hog.orientations = ori
                hog.pixels_per_cell = (ppc, ppc)
                hog.cells_per_block = (cpb, cpb)
                
                # nastaveni nazvu slozky
                childname_hog = "ori="+str(ori)+"_ppc="+str(ppc)+"_cpb="+str(cpb)
                
                # spocteni velikocti fv
                fvlp = ori * cpb**2 * ( (hog.sliding_window_size[0] // ppc) - (cpb - 1) )**2
                print "Predpokladana velikost feature vektoru: ", fvlp
                # pokud bude fv moc dlouhy, tak fitnout jen na casti a pak transformovat kazdy
                partially = fvlp >= 2000
                # pokud budou male vektory, tak muzeme extrahovat 
                # originalni data a tim padem nechceme PCA provadet u extrakce
                hog.PCA_mode = partially
                
                #decompositions = decompositions[:1]

                if partially:
                    # zatim nastavim PCA_mode v extractoru na False -> 
                    #        -> to fitnuti si totiz udelam sam
                    hog.PCA_mode = False
                    # natrenovani metody
                    decompositions = tester.fit_methods(positives, 
                                                        negatives, 
                                                        decompositions)
                    # ted uz nastavim PCA mode na True, aby mi to vyhazovalo
                    #     transformovane vektory podle dane metody
                    hog.PCA_mode = True
                    # testovani                        
                    for decomposition in decompositions:                       
                        # zjisteni velikosti redukovaneho vektoru
                        dec_name = tester.get_methodname(decomposition)
                        # nastaveni metody redukce dimenze
                        hog.PCA_object = decomposition
                        # doplneni nazvu slozky
                        tester.childname = childname_hog + "_" + dec_name
                        # extrakce jiz transformovanych dat
                        X, y = tester.extract_data(positives, negatives)
                        # cross validace
                        tester.cross_validation(X, y)
                        
                        # vypsani informace o progresu
                        iters += 1
                        print "Time: ", time.time() - t, 
                        print " - ", iters, " z ", max_iters
                        
                else:
                    # extrakce originalnich feature vektoru
                    X_raw, y = tester.extract_data(positives, negatives)
                    print "[INFO] Data shape: ", X_raw.shape
                    # testovani 
                    for decomposition in decompositions:
                        # zjisteni velikosti redukovaneho vektoru
                        dec_name = tester.get_methodname(decomposition)
                        # doplneni nazvu slozky
                        tester.childname = childname_hog + "_" + dec_name
                        # redukce dimenzionality na celych datech
                        X = decomposition.fit_transform(X_raw, y)
                        # cross validace
                        tester.cross_validation(X, y)
                        
                        # vypsani informace o progresu
                        iters += 1
                        print "Time: ", time.time() - t, 
                        print " - ", iters, " z ", max_iters
    
    #dr.save_obj((X_raw, y), tester.parentname+"data.pklz")
    
    # zaloha vysledku
    tester.backup_test_results(positives, negatives)
    
    print "Celkovy cas: ", time.time() - t     

           
    """ obsolete """ 

#    # prebarvovani
#    colorings = [None, 25, 29, 33]
#    
#    # preprocessing
#    processing_methods = [(cv2.bilateralFilter, [[9, 35, 35]]),
#                          (cv2.medianBlur, [7, 9, 11, 13])]

#    for coloring in colorings:
#        # rozmyslet si, zda to nebudu testovat rucne
#        hog.background_coloring_ksize = coloring
#        if coloring is None:
#            hog.to_color = False
#            

              
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
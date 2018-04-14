# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 10:36:02 2018

@author: Mirab
"""

import feature_extractor as fe
import classifier as clas

import sys


def testing(svm, to_train=False, to_evaluate=True, to_test=True, to_hnm=True):
    """ Otestuje klasifikator SVM s vyuzitim HoG fetaures """

    if to_hnm:
        HNM(svm, train_before=True)
        
    if to_train and not to_hnm:
        svm.create_training_data()
        svm.train()
    
    if to_test:
        svm.classify_test_images(visualization=bool(0),
                                 final_visualization=bool(0),
                                 to_print=bool(0))
                                 
    if to_evaluate:
        svm.evaluate_nms_results_overlap(print_steps=bool(1), 
                                         orig_only=bool(0))
        svm.store_results()
    
    return svm.data, svm.labels


def HNM(svm, train_before=False, train_after=True):
    """ Provede Hard negative mining """
    
    # kdyby bylo nutne pretrenovat
    if train_before:
        svm.create_training_data()
        svm.train()
    
    svm.data = None
    svm.labels = None
    svm.test_results = {}
    svm.extractor.features = {}
    
    # specifikace, na kterych datech ma byt HNM provedeno
    origs = svm.config["HNM_positives"]
    
    # hard negative mining
    svm.hard_negative_mining(visualization=bool(0),
                             final_visualization=False, 
                             origs=origs)
    # pretrenovani po HNM  
    if train_after:
        svm.create_training_data()
        svm.train()


if __name__ =='__main__':
    
    to_hnm = False
    to_train = False
    to_test = False
    to_evaluate = False
    
    if len(sys.argv) >= 2:
        for arg in sys.argv[1:]:
            if "hnm" in arg:
                to_hnm = True
            if "train" in arg:
                to_train = True
            if "test" in arg:
                to_test = True
            if "to_evaluate" in arg:
                to_evaluate = True
    else:
        to_train = True                
    
    print "--- HoG ---"
    # feature extractor
    ext = fe.HOG()
    print "HoG parametry: ", ext.orientations, ext.pixels_per_cell, ext.cells_per_block
    
    print "--- SVM ---"
    # klasifikator
    svm = clas.Classifier(extractor = ext)
    svm.extractor.load_PCA_object()
    
    svm.dataset.log_info("- - - - - - - - - - - - - - - - - - - -")
    svm.dataset.log_info("_________ test_classifiers.py _________")
    
    """ Metody ke spusteni """
    testing(svm, 
            to_hnm = to_hnm,
            to_train=to_train,
            to_evaluate=to_evaluate,
            to_test=to_test)         # klasifikace na testovacich datech
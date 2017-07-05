# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 21:26:27 2017

@author: mira
"""

import data_reader as dr
import feature_extractor as fe
import classifier as clas

"""
# SIFT
sift = fe.SIFT()

svm = clas.Classifier(extractor = sift)
svm.create_training_data()

TM = svm.data
tl = svm.labels
t = TM
l = tl

svm.train()

svm.classify_test_images()
"""



if __name__ =='__main__': 
    
    """ Otestuje klasifikator SVM s vyuzitim HoG fetaures """
    print "--- HoGy ---"
    ext = fe.HOG()
#    ext = fe.SIFT()
#    ext = fe.SURF()
    
    svm = clas.Classifier(extractor = ext)
    svm.create_training_data()
    
    TM = svm.data
    tl = svm.labels
    t = TM
    l = tl
    
    svm.train()
    
    #svm.classify_test_images(visualization=True)
    svm.hard_negative_mining(visualization=True)
    
    
    
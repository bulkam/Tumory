# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 21:26:27 2017

@author: mira
"""

import feature_extractor as fe
import classifier as clas


def obsolete():
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
    
    
def NMS(svm):
    """ Provede Non-maxima suppression pro dany vysledek testu """
    
    #svm.non_maxima_suppression("datasets/processed/test_images/00_copy_of_180_arterial-GT010.pklz")
    svm.non_maxima_suppression("datasets/processed/test_images/183a_venous-GT018.pklz")


def testing(svm, to_train=True):
    """ Otestuje klasifikator SVM s vyuzitim HoG fetaures """
    
    svm.create_training_data()
    
    TM = svm.data
    tl = svm.labels
    
    if to_train:
        svm.train()
    
    svm.classify_test_images(visualization=False,
                             final_visualization=True,
                             to_print=False)
    
    #store_results(svm)
    
    return TM, tl


def HNM(svm, to_train=False):
    """ Provede Hard negative mining """
    
    # kdyby bylo nutne pretrenovat
    if to_train:
        svm.create_training_data()
        svm.train()

    svm.hard_negative_mining(visualization=bool(0),
                             final_visualization=False)
    # pretrenovani po HNM                         
    svm.create_training_data()
    svm.train()


if __name__ =='__main__':
    
    print "--- HoGy ---"
    ext = fe.HOG()
#    ext = fe.SIFT()
#    ext = fe.SURF()
    
    # klasifikator
    svm = clas.Classifier(extractor = ext)
    svm.dataset.log_info("- - - - - - - - - - - - - - - - - - - -")
    svm.dataset.log_info("_________ test_classifiers.py _________")
    
    """ Metody ke spusteni """
    testing(svm, to_train=bool(0))            # klasifikace na testovacich datech
#    HNM(svm, to_train=bool(0))               # Hard negative mining
    
#    NMS(svm)                  # Non-maxima suppression pro nejaky vysledek
    
    svm.store_results()

    
    
    
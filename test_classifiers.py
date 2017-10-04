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


def testing(svm, to_train=True, to_evaluate=True):
    """ Otestuje klasifikator SVM s vyuzitim HoG fetaures """
    
    svm.create_training_data()
    
    TM = svm.data
    tl = svm.labels
    
    if to_train:
        svm.train()
    
    svm.classify_test_images(visualization=bool(0),
                             final_visualization=True,
                             to_print=bool(0))
    if to_evaluate:
        svm.evaluate()
    #store_results(svm)
    
    return TM, tl


def HNM(svm, train_before=False, train_after=True):
    """ Provede Hard negative mining """
    
    
    # kdyby bylo nutne pretrenovat
    if train_before:
        svm.create_training_data()
        svm.train()
    
    # specifikace, na kterych datech ma byt HNM provedeno
    origs = svm.config["HNM_positives"]
    HNMs = svm.config["HNM_HNMs"]
    
#    origs=[30, 30]
#    HNMs=[41, 70]
    
    # hard negative mining
    svm.hard_negative_mining(visualization=bool(0),
                             final_visualization=False, 
                             origs=origs,
                             HNMs=HNMs)
    # pretrenovani po HNM  
    if train_after:
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
    testing(svm, to_train=bool(0),
            to_evaluate=bool(1))            # klasifikace na testovacich datech
#    HNM(svm, train_before=bool(0))         # Hard negative mining
    
#    NMS(svm)                  # Non-maxima suppression pro nejaky vysledek
    
    svm.store_results()

    
    
    
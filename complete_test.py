# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 21:08:34 2017

@author: mira
"""

import test_modules as tm
import test_classifiers as tc

import feature_extractor as fe
import classifier as clas



def test(to_extract=True, to_train=True, to_test=True):
    """ Pripadne provede: 
            extrakci vektoru priznaku 
            natrenovani klasifikatoru
            testovani 
    """
    
    # extrakce vektoru priznaku
    if to_extract: tm.test_hogs()
    
    # trenovani a klasifikace
    ext = fe.HOG()
    
    # klasifikator
    svm = clas.Classifier(extractor = ext)
    svm.dataset.log_info("- - - - - - - - - - - - - - - - - - - -")
    svm.dataset.log_info("_________ complete_test.py _________")
    svm.dataset.log_info("      extract: " + str(to_extract))
    svm.dataset.log_info("      train:   " + str(to_train))
    svm.dataset.log_info("      test:    " + str(to_test))
    
    """ Metody ke spusteni """
    if to_test: 
        tc.testing(svm, to_train=to_train)  # klasifikace na testovacich datech
    
    svm.store_results()
    
    svm.dataset.log_info("_________ KONEC complete_test.py _________")


if __name__ =='__main__':
    
    test(to_extract=bool(1), to_train=bool(1))
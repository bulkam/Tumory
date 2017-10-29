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
    

def multiple_test(to_hnm=False):
    """ Pripadne provede: 
            extrakci vektoru priznaku 
            natrenovani klasifikatoru
            testovani 
    """
    
    # vyfiltrovane hog konfigurace
    oris = [9, 12]
    ppcs = [6, 4]
    cpbs = [2]

    # nejvetsi data nejdrive
    oris, ppcs, cpbs = oris[::-1], ppcs[::-1], cpbs[::-1]
    
    for ori in oris:
        for ppc in ppcs:
            for cpb in cpbs:
                print [ori, ppc, cpb]
                
                # vytvoreni extraktoru
                ext = fe.HOG()
                # nastaveni parametru extraktoru
                ext.orientations = ori
                ext.pixels_per_cell = (ppc, ppc)
                ext.cells_per_block = (cpb, cpb)
                # extrakce vektoru priznaku
                ext.extract_features(to_save=bool(0), multiple_rois=bool(1), 
                                     PCA_partially=bool(1), save_features=bool(1))
                
                # klasifikator
                svm = clas.Classifier(extractor = ext)
                # zalogovani zprav
                svm.dataset.log_info("- - - - - - - - - - - - - - - - - - - -")
                svm.dataset.log_info("_________ complete_test.py -> multiple_test() _________")
                svm.dataset.log_info("            hnm: " + str(to_hnm))
                svm.dataset.log_info("            hog: " + str([ori, ppc, cpb]))
                
                """ Metody ke spusteni """
                # testovani na vsech testovacich datech
                tc.testing(svm, to_train=True)  # klasifikace na testovacich datech
                # ulozeni vysledku
                print "[INFO] Ukladam vysledky...",
                svm.store_results(suffix="median13_win48_col27_ori="+str(ori)+"_ppc="+str(ppc)+"_cpb="+str(cpb))
                print "Hotovo."
                
    svm.dataset.log_info("_________ KONEC complete_test.py _________")


if __name__ =='__main__':
    
    #test(to_extract=bool(1), to_train=bool(1))
    multiple_test()
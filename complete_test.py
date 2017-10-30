# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 21:08:34 2017

@author: mira
"""

import test_modules as tm
import test_classifiers as tc

import feature_extractor as fe
import classifier as clas

import time



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
                # vytvoreni klasifikatoru
                svm = clas.Classifier(extractor = ext)
                
                # nastaveni parametru extraktoru
                svm.extractor.orientations = ori
                svm.extractor.pixels_per_cell = (ppc, ppc)
                svm.extractor.cells_per_block = (cpb, cpb)
                
                # spocteni velikocti fv a pripadna redukce poctu dat pro PCA
                fvlp = ori * cpb**2 * ( (svm.extractor.sliding_window_size[0] // ppc) - (cpb - 1) )**2
                print "Predpokladana velikost feature vektoru: ", fvlp
                if fvlp > 2000:
                    svm.extractor.n_for_PCA = 1500
                if fvlp > 2500:
                    svm.extractor.n_for_PCA = 1000
                if fvlp > 4000:
                    svm.extractor.n_for_PCA = 700
                if fvlp > 5000:
                    svm.extractor.n_for_PCA = 500
                    
                # extrakce vektoru priznaku
                svm.extractor.extract_features(to_save=bool(0), multiple_rois=bool(1), 
                                     PCA_partially=bool(1), save_features=bool(1))
                svm.extractor.features = dict()
                
                # klasifikator
                svm.extractor.load_PCA_object()
                # zalogovani zprav
                svm.dataset.log_info("- - - - - - - - - - - - - - - - - - - -")
                svm.dataset.log_info("_________ complete_test.py -> multiple_test() _________")
                svm.dataset.log_info("            hnm: " + str(to_hnm))
                svm.dataset.log_info("            hog: " + str([ori, ppc, cpb]))
                
                """ Metody ke spusteni """
                # testovani na vsech testovacich datech
                tc.testing(svm, to_train=True)  # klasifikace na testovacich datech
                # ohodnoceni prekryti
                svm.evaluate_nms_results_overlap()
                # ulozeni vysledku
                print "[INFO] Ukladam vysledky...",
                svm.store_results(suffix="bilat9-35-35_win48_col27_ori="+str(ori)+"_ppc="+str(ppc)+"_cpb="+str(cpb))
                print "Hotovo."
                
                
                
    svm.dataset.log_info("_________ KONEC complete_test.py _________")


if __name__ =='__main__':
    
    t = time.time()
    
    #test(to_extract=bool(1), to_train=bool(1))
    multiple_test()
    
    print "[INFO] Celkovy cas:", time.time() - t
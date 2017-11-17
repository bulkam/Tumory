# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 21:08:34 2017

@author: mira
"""

import test_modules as tm
import test_classifiers as tc

import feature_extractor as fe
import classifier as clas

#import helper_test as hlt

import time
import cv2
import copy


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
            (hard negative mining)
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
                svm.dataset.log_info("     double_hnm: " + str(svm.double_HNM))
                svm.dataset.log_info("            hog: " + str([ori, ppc, cpb]))
                
                """ Metody ke spusteni """
                if to_hnm:
                    # hard negative mining predtim
                    print "[INFO] Hard negative mining..."
                    tc.HNM(svm, train_before=True)
                # testovani na vsech testovacich datech
                tc.testing(svm, to_train = not to_hnm)  # klasifikace na testovacich datech
                
                # ohodnoceni prekryti
                svm.evaluate_nms_results_overlap()
                # ulozeni vysledku
                print "[INFO] Ukladam vysledky...",
                svm.store_results(suffix="HNM=best50_median9_win48_col27_ori="+str(ori)+"_ppc="+str(ppc)+"_cpb="+str(cpb))
                print "Hotovo."
                
                
    svm.dataset.log_info("_________ KONEC complete_test.py _________")

def bilateral9(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.bilateralFilter(out, 9, 35, 35)
    
def bilateral9_75_75(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.bilateralFilter(out, 9, 75, 75)
    
def bilateral9_55_55(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.bilateralFilter(out, 9, 55, 55)
    
def bilateral9_15_15(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.bilateralFilter(out, 9, 15, 15)
    
def bilateral13_75_75(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.bilateralFilter(out, 13, 75, 75)
    
def bilateral13_55_55(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.bilateralFilter(out, 13, 55, 55)
    
def bilateral13_15_15(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.bilateralFilter(out, 13, 15, 15)
    
def bilateral13_35_35(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.bilateralFilter(out, 13, 35, 35)

def bilateral17_75_75(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.bilateralFilter(out, 17, 75, 75)
    
def bilateral17_55_55(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.bilateralFilter(out, 17, 55, 55)
    
def bilateral17_35_35(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.bilateralFilter(out, 17, 35, 35)

def bilateral15_75_75(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.bilateralFilter(out, 15, 75, 75)
    
def bilateral15_55_55(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.bilateralFilter(out, 15, 55, 55)
    
def bilateral15_35_35(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.bilateralFilter(out, 15, 35, 35)

def bilateral7_75_75(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.bilateralFilter(out, 7, 75, 75)
    
def bilateral7_55_55(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.bilateralFilter(out, 7, 55, 55)
    
def bilateral7_35_35(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.bilateralFilter(out, 7, 35, 35)

def bilateral11_75_75(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.bilateralFilter(out, 11, 75, 75)
    
def bilateral11_55_55(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.bilateralFilter(out, 11, 55, 55)
    
def bilateral11_35_35(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.bilateralFilter(out, 11, 35, 35)

def median13(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.medianBlur(out, 13)
    
def median17(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.medianBlur(out, 17)

def median15(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.medianBlur(out, 15)

def median7(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.medianBlur(out, 7)

def median11(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.medianBlur(out, 11)
    
def median9(roi):
    out = copy.copy(roi.astype("uint8"))
    return cv2.medianBlur(out, 9)


def extra_multiple_test(to_hnm=False):
    """ Pripadne provede: 
            extrakci vektoru priznaku
            (hard negative mining)
            natrenovani klasifikatoru
            testovani 
        Navic provede test pro ruzne image processingy
    """
#    methods = {"HNM=best50_median13_NOcoloring": median13,
#               "HNM=best50_median17_NOcoloring": median17,
#               "HNM=best50_bilateral9_NOcloring": bilateral9,
#               "HNM=best50_median9_NOcoloring": median9}
    
    methods = {"HNM=best50_bilateral7_55_55_NO_coloring": bilateral7_55_55,
               "HNM=best50_bilateral7_35_35_NO_coloring": bilateral7_35_35,
               "HNM=best50_bilateral7_75_75_NO_coloring": bilateral7_75_75,
               "HNM=best50_bilateral11_55_55_NO_coloring": bilateral11_55_55,
               "HNM=best50_bilateral11_75_75_NO_coloring": bilateral11_75_75,
               "HNM=best50_bilateral11_35_35_NO_coloring": bilateral11_35_35}


#    methods = {"HNM=best50_median7_NOcoloring": median7,
#               "HNM=best50_median11_NOcoloring": median11}        
    
    for methodlabel, method in methods.items():
    
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
                    ext.apply_image_processing = method
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
                    svm.dataset.log_info("_________ complete_test.py -> extra_multiple_test() _________")
                    svm.dataset.log_info("            hnm: " + str(to_hnm))
                    svm.dataset.log_info("     double_hnm: " + str(svm.double_HNM))
                    svm.dataset.log_info("     processing: " + str(methodlabel))
                    svm.dataset.log_info("            hog: " + str([ori, ppc, cpb]))
                    
                    """ Metody ke spusteni """
                    if to_hnm:
                        # hard negative mining predtim
                        print "[INFO] Hard negative mining..."
                        tc.HNM(svm, train_before=True)
                    # testovani na vsech testovacich datech
                    tc.testing(svm, to_train = not to_hnm)  # klasifikace na testovacich datech
                    
                    # ohodnoceni prekryti
                    svm.evaluate_nms_results_overlap(print_steps=False)
                    # ulozeni vysledku
                    print "[INFO] Ukladam vysledky...",
                    svm.store_results(suffix=methodlabel+"_win48_ori="+str(ori)+"_ppc="+str(ppc)+"_cpb="+str(cpb))
                    print "Hotovo."
                    
                    
    svm.dataset.log_info("_________ KONEC complete_test.py _________")


def extra_multiple_retest():
    
    methods = {"HNM=best50_median13_NO_coloring": median13,
               "HNM=best50_median17_NO_coloring": median17,
               "HNM=best50_bilateral9_NO_coloring": bilateral9,
               "HNM=best50_median9_NO_coloring": median9}
               
    foldernames = hlt.get_foldernames()
    for foldername in foldernames:
        #print hlt.find_processing(foldername)
        methodlabel = ""
        method = ""
        ori = hlt.find_ori(foldername)
        ppc = hlt.find_ppc(foldername)
        cpb = hlt.find_cpb(foldername)
        print ori, ppc, cpb
        
    
    #print foldernames

if __name__ =='__main__':
    
    t = time.time()
    
    #test(to_extract=bool(1), to_train=bool(1))
    #multiple_test(to_hnm=True)
    extra_multiple_test(to_hnm=True)
    #extra_multiple_retest()
    
    print "[INFO] Celkovy cas:", time.time() - t
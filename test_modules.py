# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 18:24:33 2017

@author: mira
"""

import feature_extractor as fe


def test_hogs():
    hog = fe.HOG()
    path = hog.config_path
    config = hog.dataset.config
    win = hog.sliding_window_size
    
    data = hog.dataset
    data.log_info("- - - - - - - - - - - - - - - - - - - -")
    data.log_info("_________ test_modules.py _________")
    
    print hog.sliding_window_size
    print len(data.orig_images)
    
    # spocteni velikocti fv a pripadna redukce poctu dat pro PCA
    ori = hog.orientations
    ppc = hog.pixels_per_cell[0]
    cpb = hog.cells_per_block[0]
    fvlp = ori * cpb**2 * ( (hog.sliding_window_size[0] // ppc) - (cpb - 1) )**2
    print "Predpokladana velikost feature vektoru: ", fvlp
    if fvlp > 2000:
        hog.n_for_PCA = 1500
    if fvlp > 2500:
        hog.n_for_PCA = 1000
    if fvlp > 4000:
        hog.n_for_PCA = 700
    if fvlp > 5000:
        hog.n_for_PCA = 500
    #print hog.dataset.load_annotated_images()
    TM = hog.extract_features(to_save=bool(1), multiple_rois=bool(1), 
                              PCA_partially=bool(1), save_features=bool(0))
    
    return TM


def test_others():
    sift = fe.SURF()
    path = sift.config_path
    config = sift.dataset.config
    win = sift.sliding_window_size
    
    data = sift.dataset
    
    #print hog.dataset.load_annotated_images()
    TM = sift.extract_features()
    
    return TM


if __name__ =='__main__':
    
    TM = test_hogs()
    #t = TM
    
    #TM = test_others()
    
    #for each in TM.keys():
    #    if not each.startswith("neg"):
    #        print each, len(TM[each]["feature_vect"]), TM[each]["label"]
    #    if each.startswith("neg"):
    #        print each, len(TM[each]["feature_vect"]), TM[each]["label"]

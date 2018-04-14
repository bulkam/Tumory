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
    data.log_info("_________ extract_feature_vectors.py _________")
    
    print hog.sliding_window_size
    print len(data.orig_images)
    
#    print hog.count_sliding_window_size(data.config["annotations_path"])
    
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
    
    # extrakce vektoru priznaku
    TM = hog.extract_features(to_save=bool(0), multiple_rois=bool(1), 
                              PCA_partially=bool(1), save_features=bool(1))
    
    return TM


if __name__ =='__main__':
    
    TM = test_hogs()

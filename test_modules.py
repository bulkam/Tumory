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
    #print hog.dataset.load_annotated_images()
    TM = hog.extract_features(to_save=bool(0), multiple_rois=bool(1), PCA_partially=bool(1))
    
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
    t = TM
    
    #TM = test_others()
    
    #for each in TM.keys():
    #    if not each.startswith("neg"):
    #        print each, len(TM[each]["feature_vect"]), TM[each]["label"]
    #    if each.startswith("neg"):
    #        print each, len(TM[each]["feature_vect"]), TM[each]["label"]

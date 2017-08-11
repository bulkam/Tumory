# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:34:18 2017

@author: mira
"""

import data_reader as dr


def analyze_training_data(data):
    """ Analyza trenovacich dat """
    
    print "Total:",len(data.keys())
    pos = 0
    neg = 0
    
    for key in data.keys():
        if "neg" in key:
            neg += 1
        else:
            pos += 1
    
    print "Pos:  ", pos
    print "NEG:  ", neg


def analyze_features(dataset, data):
    """ Analyza delky vektoru priznaku """
    
    expected = dataset.config["feature_vector_length"]
    print "Expected:", expected
    
    i = 0
    for key in data.keys():
        if not len(data[key]["feature_vect"]) == expected:
            #print "[ERROR] ", key, " - delka: ", len(data[key]["feature_vect"])
            i += 1
    
    print "Celkem ", i, " vektoru nema pozadovanou delku ", expected, "!"


def make_pngs(dataset):
    """ Spusti metodu make_pngs z tridy DATAset """
    
    #dataset.make_pngs("datasets/frames/positives/")
    dataset.make_pngs("datasets/processed/hard_negative_mining/")


if __name__ =='__main__':
    
    dataset = dr.DATAset()
    
    # nacteni trenovacich dat
    path = dataset.config["training_data_path"]+"hog_features.json"
    TM = dataset.precti_json(path)
    
    # analyze trenovacich dat
    analyze_training_data(TM)
    analyze_features(dataset, TM)
    
    # vytvoreni PNG obrazku z dane slozky s obrazovymi daty
#    make_pngs(dataset)
    
    

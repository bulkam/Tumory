# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:34:18 2017

@author: mira
"""

import data_reader as dr
import numpy as np
import matplotlib.pyplot as plt


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


def analyze_box_sizes(dataset):
    boxes = dr.load_json(dataset.annotations_path)
    
    real_boxes = list()
    for key in [name for name in boxes.keys() if not "AFFINE" in name]:
        for box in boxes[key]:
            real_boxes.append(box)
    print len(real_boxes)
    
    widths = []
    heights = []
    sizes = []
    ratios = []
    
    for box in real_boxes:
        y, h, x, w = box
        width = w-x
        height = h-y
        widths.append(width)
        heights.append(height)
        sizes.append(width * height)
        ratios.append(float(width) / height)
    
    heights = np.hstack(heights)
    widths = np.hstack(widths)
    sizes = np.hstack(sizes)
    ratios = np.hstack(ratios)
    
    plt.figure()
    plt.hist(heights, bins=50)
    #plt.plot(np.histogram(heights, bins=50)[1][:50], np.histogram(heights, bins=50)[0])
    plt.title("Histogram vysek")
    plt.grid()
    plt.show()
    
    plt.figure()
    plt.hist(widths, bins=50)
    plt.title("Histogram sirek")
    plt.grid()
    plt.show()
    
    plt.figure()
    plt.hist(sizes, bins=50)
    plt.title("Histogram obsahu")
    plt.grid()
    plt.show()
    
    plt.figure()
    plt.hist(ratios, bins=50)
    plt.title("Histogram pomeru")
    plt.grid()
    plt.show()

    
if __name__ =='__main__':
    
    dataset = dr.DATAset()
    
    # nacteni trenovacich dat
    path = dataset.config["training_data_path"]+"hog_features.json"
    TM = dataset.precti_json(path)
    
    # analyza trenovacich dat
    analyze_training_data(TM)
    analyze_features(dataset, TM)
    
    # vytvoreni PNG obrazku z dane slozky s obrazovymi daty
#    make_pngs(dataset)
    
    analyze_box_sizes(dataset)
    

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:25:29 2018

@author: mira
"""

import time
import numpy as np
import cv2
import skimage.morphology


def accuracy_per_pixel(test_labels, test_predictions, 
                       truth_class=1, predicted_class=1):
    
    t = time.time()
    print(test_labels.shape, test_predictions.shape)
    
    i = truth_class
    j = predicted_class
    
    N = np.count_nonzero(test_labels[:, :, :, i] == 1)
    P1 = np.sum(test_predictions[:, :, :, j][(test_labels[:, :, :, i] == 1)])
    
    print("[INFO] Pixel volume lezi v obrazech: ", N)
    print("[INFO] Celkem ppsti v techto regionech: ", P1)
    print("[INFO] Accuracy per pixel:", P1/N)
    s = test_labels.shape
    
    print("[INFO] Zastoupeni lezi v obraze: ", float(N) / (s[0]*s[1]*s[2]*s[3]))
    print(time.time() - t) # trva to 84 sekund pro 3402 obrazku
    return P1/N
    
    
def accuracy_matrix(test_labels, test_predictions):
    
    t = time.time()
    
    classes = range(test_labels.shape[-1])
    A = np.zeros((len(classes), len(classes)))
    
    for i in classes:
        for j in classes:
            A[i,j] = accuracy_per_pixel(test_labels, test_predictions, 
                                        truth_class=i, predicted_class=j)
    
    print(A)
    print("Vysledny cas:", time.time() - t)
    
    return A


def Jaccard_similarity(binary_img, binary_ref, print_results=False):
    """ Spocita IoU vysledku a refenercniho obrazku - anotace """

    x1, y1, x2, y2 = cv2.boundingRect(binary_ref.astype("uint8"))
    xr, yr, wr, hr = x1, y1, x1+x2, y1+y2
    
    x1, y1, x2, y2 = cv2.boundingRect(binary_img)
    xi, yi, wi, hi = x1, y1, x1+x2, y1+y2
    
    if np.sum((binary_img & binary_ref).astype("uint8")) == 0:
        return 0
    
    yc = np.max((yr, yi))
    hc = np.min((hr, hi))
    xc = np.max((xr, xi))
    wc = np.min((wr, wi))
    
    intersection = (hc - yc) * (wc - xc)
    unification = ((hr - yr) * (wr - xr)) + ((hi - yi) * (wi - xi)) - intersection
    JS = float(intersection) / unification
    
    if print_results:
        print("[RESULT] Jaccard similarity: ", JS)
        
    return JS


def apply_morphology_operations(img, ref, intensity_scale=127, label_color=255):
    
    L = 1 * intensity_scale
    binary_img = img == L
    binary_ref = ref[:, :, 1] == label_color

    binary_img = cv2.morphologyEx(binary_img.astype("uint8"), 
                                  cv2.MORPH_CLOSE, 
                                  cv2.getStructuringElement(cv2.MORPH_RECT,(11,11)))
    
    binary_img = skimage.morphology.remove_small_objects(binary_img==1, min_size=64).astype("uint8")
    
    return binary_img, binary_ref


def evaluate_boxes_overlap(img, label, J_thr = 0.8, print_steps=True):
    
    binary_img, binary_ref = apply_morphology_operations(img, label)
    
    ret_img, markers_img = cv2.connectedComponents(binary_img)
    ret_lab, markers_lab = cv2.connectedComponents((binary_ref).astype("uint8"))
    
    # Projizdeni objektu a pocitani JS (IoU)
    pairs = []
    for i in range(1, ret_img):
        predicted_area = (markers_img == i).astype("uint8")
        maxJ = 0
        max_label = 0
        for j in range(1, ret_lab):
            label_area = (markers_lab == j).astype("uint8")
            JS = Jaccard_similarity(predicted_area, label_area)
            if JS > maxJ:
                maxJ = JS
                max_label = j
        pairs.append([i, max_label, maxJ])

    # Pocitani Confusion matrix
    TP, TN, FP, FN = 0, 0, 0, 0

    for j in range(1, ret_lab):
        JSims = [pair[2] for pair in pairs if pair[1] == j]
        #print JSims

        if len(JSims) == 0:
            FN += 1

        elif len(JSims) >= 2:
            FP += len(JSims) - 1
            if max(JSims) >= J_thr:
                TP += 1
            else:
                FP += 1
                FN += 1

        elif len(JSims) == 1:
            if JSims[0] >= J_thr:
                TP += 1
            else:
                FP += 1
                FN += 1
                
    if print_steps:
        print(TP,"", TN, "", FP, "", FN)
    
    return TP, TN, FP, FN


def evaluate_JS(test_labels, test_predictions, mode="argmax", Pmin=0.33):
    
    print("TP|TN|FP|FN")

    TPs, TNs, FPs, FNs = 0, 0, 0, 0

    for index in range(test_predictions.shape[0]):

        if index % 500 == 0:
            print(index, " / ", test_predictions.shape[0])

        label = test_labels[index].astype("uint8") * 255
        result = test_predictions[index].astype("float")
        
        if mode == "Pmin":
            lesion = (result[:, :, 1] >= Pmin).astype("uint8")
            
        else:
            lesion = np.argmax(result, axis=2)*127

        TP, TN, FP, FN = evaluate_boxes_overlap(lesion, label, 
                                                J_thr = 0.8,
                                                print_steps=bool(0))
        TPs += TP
        TNs += TN
        FPs += FP
        FNs += FN
        
        
    precision = float(TPs) / (TPs + FPs)
    accuracy = float(TPs + TNs) / (TPs + TNs + FPs + FNs)
    recall = float(TPs) / (TPs + FNs)
    FPC = float(FPs) / test_predictions.shape[0]

    print("TP: ", TPs)
    print("FP: ", FPs)
    print("FN: ", FNs)
    print("_______________________")
    print("Recall:    ", recall)
    print("Precision: ", precision)
    print("FPC:       ", FPC)
    
    vocab = {"precision": precision,
             "FPC": FPC,
             "recall": recall,
             "TPR": recall}
    return vocab
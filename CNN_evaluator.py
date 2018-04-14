# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:25:29 2018

@author: mira
"""

import time
import numpy as np
import cv2
import skimage.morphology


def get_one_hot_vectors(test_predictions):
    """ Vrati one hot vektory misto softmax vektoru """
    
    classes = np.argmax(test_predictions, axis=3)
    shape = test_predictions.shape
    del test_predictions
    
    one_hot_predictions = np.zeros(shape)
    n_classes = shape[-1]
    
    for i in range(n_classes):
        one_hot_vector = np.zeros(n_classes, dtype=np.bool)
        one_hot_vector[i] = True
        one_hot_predictions[classes==i] = one_hot_vector.copy()
    
    return one_hot_predictions


def accuracy_per_pixel(test_labels, test_predictions, 
                       truth_class=1, predicted_class=1):
    """ Spocita accuracy per pixel """
    
    #t = time.time()
    print(test_labels.shape, test_predictions.shape)
    
    i = truth_class
    j = predicted_class
    
    N = np.count_nonzero(test_labels[:, :, :, i] == 1)
    P1 = np.sum(test_predictions[:, :, :, j][(test_labels[:, :, :, i] == 1)])
    
#    print("[INFO] Pixel volume lezi v obrazech: ", N)
#    print("[INFO] Celkem ppsti v techto regionech: ", P1)
#    print("[INFO] Accuracy per pixel:", float(P1)/N)
#    s = test_labels.shape
#    
#    print("[INFO] Zastoupeni lezi v obraze: ", float(N) / (s[0]*s[1]*s[2]*s[3]))
#    print(time.time() - t) # trva to 84 sekund pro 3402 obrazku
    return float(P1)/N

    
    
def accuracy_matrix(test_labels, test_predictions, mode="soft", batch_size=50):
    """ Spocte accuracy matrix pro vysledky predikce site 
    porovnane s anotacemi"""    
    
    # jen prepsani vystupu na one-hot vektory
    one_hot_mode = mode in ["one-hot", "onehot", "one_hot"]

    t = time.time()
    
    classes = range(test_labels.shape[-1])
    A = np.zeros((len(classes), len(classes)))
    
    if one_hot_mode:
        C = np.zeros((len(classes), len(classes)))
        B = np.zeros((len(classes), len(classes)))
        
        for k in range(test_labels.shape[0]):
            l = np.argmax(test_labels[k], axis=2)
            p = np.argmax(test_predictions[k], axis=2)
            for i in classes:
                for j in classes:
                    C[i,j] += float(np.sum((p==j)&(l==i)))
                    B[i,j] += np.sum(l==i)
        A = C/B
        
    else:
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


def apply_morphology_operations(img, ref, intensity_scale=127, label_color=255,
                                element_closing_size=3, min_object_size=100, 
                                closing=True, from_config=False, config={}):
    """ Aplikuje morfologicke operace na vystup site """
    
    # prepsani parametru - pouzije se behem testovani morfologie   
    if from_config:
        if "closing" in config:
            closing = config["closing"]
        if "min_object_size" in config:
            min_object_size = config["min_object_size"]
        if "element_closing_size" in config:
            element_closing_size = config["element_closing_size"]
            
    L = 1 * intensity_scale
    binary_img = img == L
    binary_ref = ref[:, :, 1] == label_color
    
    # uzavreni
    if closing:
        element_closing_size_tuple = (element_closing_size, element_closing_size)
        binary_img = cv2.morphologyEx(binary_img.astype("uint8"), 
                                      cv2.MORPH_CLOSE, 
                                      cv2.getStructuringElement(cv2.MORPH_RECT, 
                                                   element_closing_size_tuple))
    
    binary_img = skimage.morphology.remove_small_objects(binary_img==1, 
                                    min_size=min_object_size).astype("uint8")
    
    return binary_img, binary_ref


def evaluate_boxes_overlap(img, label, J_thr=0.5, print_steps=True,
                           from_config=False, config={}):
    
    binary_img, binary_ref = apply_morphology_operations(img, label,
                                                         from_config=from_config,
                                                         config=config)
    
    ret_img, markers_img = cv2.connectedComponents(binary_img)
    ret_lab, markers_lab = cv2.connectedComponents((binary_ref).astype("uint8"))
    
    TP, TN, FP, FN = 0, 0, 0, 0
    maxJs = {"Js": [],
             "ret_img": ret_img,
             "ret_lab": ret_lab}
    # Projizdeni objektu a pocitani JS (IoU)
    for j in range(1, ret_lab):
        maxJ = 0
        for i in range(1, ret_img):
            predicted_area = (markers_img == i).astype("uint8")
            label_area = (markers_lab == j).astype("uint8")
            JS = Jaccard_similarity(predicted_area, label_area)
            if JS > maxJ:
                maxJ = JS
        if maxJ >= J_thr:
            TP += 1
        maxJs["Js"].append(maxJ)
            
    # Pocitani Confusion matrix
    FP += ret_img - 1 - TP   
    FN += ret_lab - 1 - TP              
    if print_steps:
        print(TP,"", TN, "", FP, "", FN)
    
    return TP, TN, FP, FN, maxJs


def re_evaluate_JS(maxJs, print_steps=False, J_thr=0.8, 
                   print_final_results=False):
    
    if print_final_results: 
        print("TP|TN|FP|FN")

    TPs, TNs, FPs, FNs = 0, 0, 0, 0
    for index in range(len(maxJs)):

        if index % 500 == 0 and print_steps and print_final_results:
            print(index, " / ", len(maxJs))
        
        Js = maxJs[index]["Js"]
        ret_img = maxJs[index]["ret_img"]
        ret_lab = maxJs[index]["ret_lab"]
        
        # projizdeni Js hodnot
        TP = 0
        for J in Js:
            if J >= J_thr:
                TP += 1
        # pricteni ke globalnim TP, FP, FN, TN hodnotam      
        TPs += TP
        FPs += ret_img - 1 - TP   
        FNs += ret_lab - 1 - TP 
        
    precision = float(TPs) / (TPs + FPs)
    accuracy = float(TPs + TNs) / (TPs + TNs + FPs + FNs)
    recall = float(TPs) / (TPs + FNs)
    FPC = float(FPs) / len(maxJs)
    
    if print_final_results:
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
        

def evaluate_JS(test_labels, test_predictions, mode="argmax", Pmin=0.33, 
                print_steps=False, J_thr=0.5, save_Js=False,
                from_config=False, config={}):
    
    print("TP|TN|FP|FN")

    TPs, TNs, FPs, FNs = 0, 0, 0, 0
    maxJs = list()

    for index in range(test_predictions.shape[0]):

        if index % 500 == 0:
            print(index, " / ", test_predictions.shape[0])

        label = test_labels[index].astype("uint8") * 255
        result = test_predictions[index].astype("float")
        
        if mode == "Pmin":
            lesion = (result[:, :, 1] >= Pmin).astype("uint8")
            
        else:
            lesion = np.argmax(result, axis=2)*127

        TP, TN, FP, FN, maxJ = evaluate_boxes_overlap(lesion, label, 
                                                      J_thr=J_thr,
                                                      print_steps=print_steps,
                                                      from_config=from_config,
                                                      config=config)
        TPs += TP
        TNs += TN
        FPs += FP
        FNs += FN
        
        if save_Js:
            maxJs.append(maxJ)
        
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
             
    if not save_Js:
        return vocab
    else:
        return vocab, maxJs
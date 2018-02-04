# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 20:36:34 2018

@author: Mirab
"""

import numpy as np
import cv2
from skimage.morphology import label as sklabel
import CNN_evaluator


def ellipse(c, smaller_scale=0.6):
    """ Vytvori masku elipsy """
    
    inv = 1.0 / smaller_scale
    rx, ry = c * smaller_scale
    
    x, y = np.ogrid[-rx*inv: rx*inv+1, -ry*inv: ry*inv+1]
    return  (x.astype(float)/rx)**2 + (y.astype(float)/ry)**2 <= 1
    

def get_mask_frame(mask, bounding_box):
    """ Z daneho scalu a souradnic exrahuje okenko masky """
    
    (x, h, y, w) = bounding_box
    return mask[x:h, y:w]


def artefact_coverage(mask_frame):
    """ Vrati zastoupeni nalezu ve snimku """
    
    # spocteni pixelu
    total_pixels = mask_frame.shape[0] * mask_frame.shape[1]
    liver_pixels = np.sum((mask_frame == 1).astype(int))
    # spocteni pokryti obrazku jatry
    return float(liver_pixels) / total_pixels


def artefact_center_ellipse_coverage(mask_frame, smaller_scale=0.6):
    """ Vytvori presne uprostred framu oblast ve tvaru elipsy,
    a vrati zastoupeni artefaktu uvnitr """
    
    # urceni rozmeru masky
    c = np.array(mask_frame.shape) // 2
    # vytvoremi masky elipsy
    ellipse_mask = ellipse(c, smaller_scale=smaller_scale)
    # zprava velikosti podle masky frmu
    ellipse_mask = cv2.resize(ellipse_mask.astype("uint8"), mask_frame.shape[::-1], interpolation = cv2.INTER_CUBIC)
    
    # vytazeni pozadovane oblasti z masky framu
    mask_ellipse_frame = mask_frame[ellipse_mask==True]
    
    # vypocet zastoupeni jater v oblasti
    total = np.sum(ellipse_mask >= 1).astype(int)
    artefact = np.sum(mask_ellipse_frame == 1).astype(int)
    coverage = float(artefact) / total
    
    return coverage, ellipse_mask
    
    
def covered_by_artefact(mask_frame):
    """ Vrati indikator, zda je box vyplnen artefaktem ci nikoliv """

    # vypocet pokryti boxu a jeho stredu artefaktem
    bb_artefact_coverage = artefact_coverage(mask_frame)
    bb_artefact_center_coverage, _ = artefact_center_ellipse_coverage(mask_frame)
    #print("COV:", bb_artefact_coverage)
    # nastaveni prahu
    # TODO: cist z configu
    min_ac = 0.4    # minimalni pokryti boxu artefaktem
    min_acc = 0.6   # minimalni pokryti stredu boxu artefaktem
    # vrati logicky soucin techto dvou podminek
    return bb_artefact_coverage >= min_ac and bb_artefact_center_coverage >= min_acc


def get_boxes_from_prediction(img, ret, padding=10):
    
    boxes = list()
    
    for i in range(1,ret):
        
        idxs = np.where(img == i)
        
        y = max(min(idxs[0])-padding, 0)
        h = min(max(idxs[0])+padding, img.shape[0])
        x = max(min(idxs[1])-padding, 0)
        w = min(max(idxs[1])+padding, img.shape[1])
        
        boxes.append([y,h,x,w])
        
    return boxes
        
    
def get_boxes(test_predictions, test_labels, padding=10, 
              mode="argmax", Pmin=0.5):
    
    bounding_boxes = list()
    
    for i in range(test_predictions.shape[0]):
 
        label = test_labels[i].astype("uint8") * 255
        result = test_predictions[i].astype("float")
        if mode == "Pmin":
            lesion = (result[:, :, 1] >= Pmin).astype("uint8")
        else:
            lesion = np.argmax(result, axis=2)*127

        binary_img, binary_ref = CNN_evaluator.apply_morphology_operations(lesion, label)

        ret_img, markers_img = cv2.connectedComponents(binary_img)
        ret_lab, markers_lab = cv2.connectedComponents((binary_ref).astype("uint8"))
        
        boxes = get_boxes_from_prediction(markers_img, ret_img, padding=10)
        bounding_boxes.append(boxes)

        
    return bounding_boxes
    

def evaluate_nms_results_overlap(test_data, test_labels, test_predictions,
                                 print_steps=False, orig_only=False,
                                 mode="argmax" ,Pmin=0.5,
                                 sliding_window_size=(48, 48)):
    """ Ohodnoti prekryti vyslednych bounding boxu s artefakty """

    # inicializace statistik
    TP, TN, FP, FN = 0, 0, 0, 0

    problematic = list()
    bounding_boxes = get_boxes(test_predictions, test_labels, padding=10)
    #print(bounding_boxes)

    for index in range(test_labels.shape[0]): 

        mask = test_labels[index].astype("uint8") * 255
        
        boxes = bounding_boxes[index]   
        mask = np.argmax(mask, axis=2)#*127


        TP0, TN0, FP0, FN0 = 0, 0, 0, 0
        
        # oriznuti obrazku a masky -> takhle se to dela u augmentovanych
        #img, mask = fe.cut_image(orig, mask)
        #mask /= 127.0
        # olabelovani artefaktu
        imlabel = sklabel(mask)
        # obarveni mist bez artefaktu na 0
        imlabel[(mask==0) | (mask==2)] = 0
        # vytvoreni prazdneho obrazku
        blank = np.zeros(mask.shape)
        # ziskani indexu artefaktu
        artefact_ids = np.unique(imlabel)[1:]
        # seznam boxu, ktere pokryvaji nejaky artefakt
        covered_box_ids = list()
        
        # prochazeni vsech artefaktu
        for i in artefact_ids:

            covered_by_bb = False

            for j, (y, h, x, w) in enumerate(boxes):
                # obarveni oblasti boxu
                blank[y:h, x:w] = 1
                # vypocet pixelu artefaktu celeho a v boxu
                na = np.sum((imlabel==i).astype(int))
                nab = np.sum((imlabel==i) & (blank==1))
                # vypocet zastoupeni bb v artefaktu
                artefact_bb_coverage = float(nab)/na

                # pokud je artefakt alespon z poloviny pokryt boxem
                if artefact_bb_coverage >= 0.5:
                    #print artefact_bb_coverage

                    covered_box_ids.append(j)
                    # vytazeni frmau masky
                    mask_frame = mask[y:h, x:w]
                    # pokud jsou pokryty artefaktem -> TP, jinak FP
                    if covered_by_artefact(mask_frame):
                        TP += 1
                        TP0 += 1
                        covered_by_bb=True
                        break
                    elif artefact_bb_coverage == 1.0 and (np.sum((blank==1).astype(int)) == sliding_window_size[0]**2):
                        TP += 1
                        TP0 += 1
                        covered_by_bb=True 
                        break
                    else:
                        FP += 1
                        FP0 += 1
                # znovu prebarveni pomocneho framu zpatky na 0      
                blank[y:h, x:w] = 0

            # pokud neni pokryt zadnym boxem alespon z poloviny
            if not covered_by_bb:# and na >= 300: # navic by mel byt dostatecne velky
                FN += 1
                FN0 += 1

        # prochazeni zatim neprohlendutych boxu
        for j in range(len(boxes)):
            if not j in covered_box_ids:
                # vytazeni boxu
                y, h, x, w = boxes[j]
                mask_frame = mask[y:h, x:w]
                # pokud jsou pokryty artefaktem -> TP, jinak FP
                if covered_by_artefact(mask_frame):
                    TP += 1
                    TP0 += 1
                else:
                    FP += 1
                    FP0 += 1

#            if FN0 > TP0:
#                print imgname
#                problematic.append(imgname)

        if print_steps: print(TP0, TN0, FP0, FN0)

    # finalni vyhodnoceni
    recall = float(TP) / (TP + FN) if TP + FN > 0 else 0
    precision = float(TP) / (TP + FP) if TP + FP > 0 else 0
    FPC = float(FP) / test_data.shape[0] if test_data.shape[0] > 0 else 0
#    if orig_only:
#        FPC = float(FP) / len([k for k in test_results_nms.keys() if not "AFFINE" in k])

    print("[RESULT] Celkove vysledky pro "+str(test_data.shape[0])+" obrazku:")
    print("         TP:", TP)
    print("         TN:", TN)
    print("         FP:", FP)
    print("         FN:", FN)
    print("        TPR:", recall)
    print("  precision:", precision)
    print("        FPC:", FPC)

    results_to_save = {"TP": TP, "TN": TN, "FP": FP, "FN": FN,
                       "TPR": recall, "recall": recall,
                       "precision": precision, "FPC": FPC,
                       "problematic": problematic}

    return TN, FP, FN, TP, results_to_save   
    
   
if __name__ =='__main__':    
    import h5py
    
    #file = h5py.File("classification/Keras/results/test_results-no_aug_20epoch_structured_data.hdf5", 'r')
    # S-liver
    file = h5py.File("classification/Keras/results/test_results-no_aug_20epoch_structured_data-liver_only.hdf5", 'r')
    # L-liver
    #file = h5py.File("classification/Keras/results/test_results-aug_5epoch_structured_data-liver_only.hdf5", 'r')
    
    test_data = file["test_data"]
    test_labels = file["test_labels"]
    test_predictions = file["test_predictions"]
    
    for key in file.keys():
        print(key)
    
    (TP, TN, FP, FN, results_as_hogs) = evaluate_nms_results_overlap(test_data, test_labels, test_predictions)
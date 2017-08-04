# -*- coding: utf-8 -*-
"""
Created on Wed May 03 18:23:02 2017

@author: mira
"""

from imtools import tools
import matplotlib.pyplot as plt

import skimage
#import skimage.segmentation as skiseg
from skimage.morphology import label

import numpy as np
import scipy
import os
import json
import copy
import glob

import pickle
import cPickle

# TODO: normalne ukladam float -> mozna pak moc velke
#       v datareaderu pak stejne pretypuji na float
def save_obj(obj, name):
    """ Ulozi data do .pkl souboru """
    
    filepath = os.path.dirname(os.path.abspath(__file__))+"/"+str(name)
    with open(filepath, 'wb') as f:
        f.write(cPickle.dumps(obj.astype("uint8"))) #obj.astype("uint8")
        f.close()


def load_obj(name):
    """ Ulozi data do .pkl souboru """
    
    filepath = os.path.dirname(os.path.abspath(__file__))+"/"+str(name)
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def precti_json(name):
    """ Nacte .json soubor a vrati slovnik """
    
    filepath = os.path.dirname(os.path.abspath(__file__))+"/"+str(name)
    mydata = {}
    with open(filepath) as d:
        mydata = json.load(d)
        d.close()
    return mydata


def zapis_json(jsondata,  name):
    """ Ulozi slovnik do .json souboru """
    
    filepath = os.path.dirname(os.path.abspath(__file__))+"/"+str(name)
    with open(filepath, 'w') as f:
        json.dump(jsondata, f)


def read_config(path="Configuration/config.json"):
    """ Nacte a vrati konfiguraci """
    
    return precti_json(path)


def clean_folder(foldername):
    """ Vymaze vsechny soubory ve slozce """
    
    foldername = os.path.dirname(os.path.abspath(__file__))+"/"+foldername+'*'
    files = glob.glob(foldername)
    
    for f in files:
        os.remove(f)
    
    print "[INFO] Veskery obsah slozky "+str(foldername)+" byl vymazan."


def clean_folders(config, section_name):
    """ Pro vsechny slozky ze seznamu vymaze jejich obsah """
    
    for folder in config[section_name]:
        clean_folder(config[folder])


def show_image_in_new_figure(img):
    """ Vykresli obrazek v novem okne """
    
    plt.figure()
    skimage.io.imshow(img, cmap = 'gray')
    plt.show()


def show_data(imgnames):
    """ Ukaze nekolik obrazku za sebou - pro kontrolu """
    
    n = min(10, len(imgnames))
    for i in xrange(n):
    
        img = load_obj("/Negatives/"+imgnames[i])
        show_image_in_new_figure(img)


def show_frame_in_image(gray, box, lw=3):
    """ Vykresli bounding box do obrazku """
    
    (y, h, x, w) = box
    
    img = copy.copy(gray)
    img[y:h, x:x+lw] = 255
    img[y:h, w-lw:w] = 255
    img[y:y+lw, x:w] = 255
    img[h-lw:h, x:w] = 255
    
    show_image_in_new_figure(img.astype(float)/255)
   

def control_images():
    """ Ukaze nekolik obrazku za sebou - pro kontrolu """
    
    imgnames_toshow = [imgname for imgname in os.listdir(os.path.dirname(os.path.abspath(__file__))+"/Negatives/") if imgname.endswith('.pklz')]
    show_data(imgnames_toshow[20:30])


def apply_CT_window(img, CT_window=[-125, 225], target_range=255, target_dtype=int):
    """ Zmeni skalu intenzit na dane CT okenko """
    
    HU_min, HU_max = CT_window
    
    img[img <= HU_min] = HU_min
    img[img >= HU_max] = HU_max
    
    img = img - HU_min
    
    CT_window_range = HU_max - HU_min
    normalization_scale = float(target_range) / CT_window_range
    
    img = (img * normalization_scale).astype(target_dtype)
    print "Novy rozsah obrazku: ", np.min(img), np.max(img)
    
    return img


def get_liver_only(img, gt_mask):
    """ Najde na obrazku olabelovana jatra 
    a vrati obrazek uvnitr jejich bounding boxu """
    
    coords = np.where(gt_mask==2)
    
    if not (len(coords[0])>0):
        return None
        
    (y, h, x, w) = min(coords[0]), max(coords[0]), min(coords[1]), max(coords[1])
    
    if not (abs(h-y)>40 and abs(x-w)>40):
        return None
    
    liver_img = copy.copy(img[y:h, x:w]) # bude tam img

    return liver_img


def get_rectangle(center, u, limit):
    """ Bez ohledu na to, v jakem uhlu je od stredu rectanglu
    nejblizsi hranicni bod, udela nasleujici:
    spocte polovinu delky hrany a podle ni vypocte souradnice ctverce
    :return: souradnice
    """
    
    cy, cx = center
    
    # vypocet poloviny delky strany
    a2 = int(u * (1.0 / np.sqrt(2)) + 0.5)
    
    # vypocet novych souradnic
    y = max(0, cy - a2)
    h = min(limit[0]-1, cy + a2)
    x = max(0, cx - a2)
    w = min(limit[1]-1, cx + a2)
    
    return (y, h, x, w)

# predtim byla minsize 40
def liver_inside_only_generator(img, gt_mask, fill_holes=False, 
                                metric='taxicab', show_frame=False,
                                min_size=60):
    """ Najde na obrazku olabelovana jatra 
    a vrati nejvetsi rectangle v nich """
    
    liver = gt_mask == 2  # dat normalnegt_mask==2, bylo tam jen >=1
    
    if fill_holes:
        liver = scipy.ndimage.binary_fill_holes(liver)
    
    while True:
        
        if not metric in ["chessboard", "taxicab"]:
            dist_to_border = scipy.ndimage.distance_transform_edt(liver)
        
        else:
            dist_to_border = scipy.ndimage.distance_transform_cdt(liver, metric=metric)
        
        max_dist = np.max(dist_to_border)
        coords = np.where(dist_to_border==max_dist)
        center = [coords[0][0], coords[1][0]]
        
        dist_to_border[center[0]] = 255
        dist_to_border[:, center[1]] = 255
        
        if not (len(coords[0])>0):
            break            
            
        (y, h, x, w) = get_rectangle(center, max_dist, img.shape)
        
        if not (abs(h-y) > min_size and abs(x-w) > min_size):
            break
        
        liver_img = copy.copy(img[y:h, x:w]) # bude tam img
        
        if show_frame:
            show_frame_in_image(img, (y, h, x, w), lw=3)

        liver[y:h, x:w] = False
        
        yield liver_img


def get_bounding_boxes(gt_mask):
    """ Vytahne z obrazku bounding boxy """
    
    mask = (gt_mask==1).astype(float)   
    #print np.unique(gt_mask)
    
    mask = scipy.ndimage.binary_fill_holes(mask)
    labeled = label(mask)

    n = np.max(labeled)
    if n == 0:
        return []
    
    boxes = list()
    for i in xrange(1, n+1):
        obj = labeled == i
        coords = np.where(obj>0.5)
        (y, h, x, w) = min(coords[0]), max(coords[0]), min(coords[1]), max(coords[1])
        boxes.append((y, h, x, w))
        
    return boxes
        

def load_CT(imgname, bounding_boxes, suffix='.pklz', HNM=True, each_to_HNM=1):
    """ Nacte CT a ulozi slice a pripadny bounding box """
    
    print "   Zpracovavam obrazek "+imgname
    data, gt_mask, voxel_size = tools.load_pickle_data(imgname)
    n_of_slices = data.shape[0]
    n_negative_slices = 0
    suffix_len = len(suffix)
    
    # Aplikace CT okenka:
    data = apply_CT_window(data)#, target_range=1.0, target_dtype=float)
    
    for i in xrange(n_of_slices):
        print i,
        
        data_slice = data[i]
        mask_slice = gt_mask[i]
        
        boxes = get_bounding_boxes(mask_slice)
        
        # pokud jsou nejake bounding boxy, mel by je vytahnout a obraz ulozit jako positives
        if len(boxes) >= 1:
            img_id = imgname[0:-suffix_len]+str("%03d" % int(i))+suffix
            # Ulozeni bounding boxu
            bounding_boxes["datasets/processed/orig_images/"+img_id] = boxes
            # Ulozeni obrazku
            #skimage.io.imsave("Slices/"+imgname[0:-suffix_len]+str("%03d" % int(i))+".png", data_slice)
            save_obj(data_slice.astype("uint8"), "Slices/"+img_id)
            save_obj(mask_slice.astype("uint8"), "Masks/"+img_id)
                
        # jinak je oznaci jako negativni
        else:
            img_id = imgname[0:-suffix_len]+str("%03d" % int(i))+suffix
            
            # kazdy nekolikaty negativni rez dat do hard_negatives
            if HNM:
                n_negative_slices += 1
                    
                if (n_negative_slices % each_to_HNM == 0):
                    img_id = imgname[0:-suffix_len]+str("%03d" % int(i))+suffix
                    skimage.io.imsave("frames/hnm/"+img_id[0:-suffix_len]+".png", data_slice)
                    save_obj(data_slice.astype("uint8"), "Hard_negative_mining/"+img_id)
                    save_obj(mask_slice.astype("uint8"), "Masks/"+img_id)
                
            # jinak je da do trenovaci mnoziny negatives
            frame_id = 0
            for data_to_save in liver_inside_only_generator(data_slice, gt_mask[i], 
                                                            fill_holes=False, 
                                                            metric='taxicab'):
                if not data_to_save is None:
                    img_id = imgname[0:-suffix_len]+str("%03d" % int(i))+"-"+str("%03d" % int(frame_id))+suffix
                    # ulozeni obrazku
                    #skimage.io.imsave("Negatives/"+imgname[0:-suffix_len]+str("%03d" % int(i))+".png", data_to_save)
                    save_obj(data_to_save.astype("uint8"), "Negatives/"+img_id)
                    frame_id += 1
            
    print ""


if __name__ =='__main__':
    
    config = read_config()
    clean_folders(config, "folders_to_clean")
    
    imgnames = [imgname for imgname in os.listdir(os.path.dirname(os.path.abspath(__file__))) if imgname.endswith('.pklz')]
    print imgnames
    bounding_boxes = dict()
    
    for imgname in imgnames:
        load_CT(imgname, bounding_boxes)
        #break
        
    # ulozeni anotaci
    #print bounding_boxes
    zapis_json(bounding_boxes, "bounding_boxes/bounding_boxes.json")
    
    """
    
    i = 0
    imgnames = [imgname for imgname in os.listdir(os.path.dirname(os.path.abspath(__file__))) if imgname.endswith('.pklz')]
    
    print "   Zpracovavam obrazek "+imgnames[i]
    data, gt_mask, voxel_size = tools.load_pickle_data(imgnames[i])
    n_of_slices = data.shape[0]
    suffix = '.pklz'
    suffix_len = len(suffix)
    
    # Aplikace CT okenka:
    data = apply_CT_window(data)#, target_range=1.0, target_dtype=float)    
    data_slice = data[15]
    #show_image_in_new_figure(data_slice.astype(float)/255)
     
    print np.unique(gt_mask)
    data_to_save = get_liver_only(data_slice, gt_mask[15])
    #show_image_in_new_figure(data_to_save.astype(float)/255)
    
    for frame in liver_inside_only_generator(data_slice, gt_mask[15], fill_holes=False, metric='taxicab', show_frame=True):
        pass
        #show_image_in_new_figure(frame.astype(float)/255)
    """



    
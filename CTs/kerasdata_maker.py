# -*- coding: utf-8 -*-
"""
Created on Mon Dec 04 08:41:33 2017

@author: mira
"""

from imtools import tools
import matplotlib.pyplot as plt

import skimage
#import skimage.segmentation as skiseg
from skimage.morphology import label
import skimage.transform as tf

import numpy as np
import cv2

import scipy
import os
import json
import copy
import glob

import pickle
import cPickle

import file_helper as fh
import slice_executer as se


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


def cut_image(img, mask):
    """ Orizne augmentovany obrazek i masku """
    
    # hledani oblasti, kde je obrazek
    relevant = np.where(img > 0)

    minh = np.min(relevant[0])
    maxh = np.max(relevant[0])
    minw = np.min(relevant[1])
    maxw = np.max(relevant[1])

    small_img = img[minh:maxh+1, minw:maxw+1]
    small_mask = mask[minh:maxh+1, minw:maxw+1]
    
    return small_img, small_mask


def resize_data(img, mask, new_shape=(232, 240)):
    
    new_img = cv2.resize(img.astype("uint8"), new_shape, interpolation = cv2.INTER_CUBIC)
    new_mask = cv2.resize(mask, new_shape, interpolation = cv2.INTER_NEAREST)
    return new_img, new_mask


def transform_data_affine(data, mask, scale=(1, 1), rotation=0, shear=0):
    """ Augmentace dat - affinni transformace """
    
    lab = "_rot="+str(rotation)+"_shear="+str(int(shear*10))
    #lab = lab + "_scale="+str(int(scale[0]*10))+"x"+str(int(scale[1]*10))
    
    # padding obrazku, aby byla rezerva a neorezavalo se to
    new_data = np.zeros((data.shape[0]*2, data.shape[1]*2))
    new_data[new_data.shape[0]//4: 3*new_data.shape[0]//4, new_data.shape[1]//4: 3*new_data.shape[1]//4] = copy.copy(data)
    new_mask = np.zeros((data.shape[0]*2, data.shape[1]*2))
    new_mask[new_mask.shape[0]//4: 3*new_mask.shape[0]//4, new_mask.shape[1]//4: 3*new_mask.shape[1]//4] = copy.copy(mask)
    
    dh, dw = np.array(new_data.shape) / 2
    # zakladni transformace
    basic = skimage.transform.AffineTransform(matrix=None, 
                                                    scale=scale, 
                                                    rotation=np.deg2rad(rotation), 
                                                    shear=shear, 
                                                    translation=None)
    # korekce posunuti
    trans = tf.AffineTransform(translation=[-dw, -dh])
    transinv = tf.AffineTransform(translation=[dw, dh])
    # vysledna transformace
    g_transform = trans + (basic + transinv)
    # aplikace transformace
    new_data = tf.warp(new_data.astype("float64"), g_transform, order=0).astype("int")
    #new_mask = (tf.warp(new_mask.astype("float64"), g_transform)+0.5).astype("int")
    new_mask = tf.warp(new_mask, g_transform, order=0).astype("int")
    # korekce
    #new_mask = remove_small_objects(new_mask)
    
    # oriznuti dat
    new_data, new_mask = cut_image(new_data, new_mask)
    # TODO: resize dat
    new_data, new_mask = resize_data(new_data, new_mask)
    
    return new_data, new_mask, lab


def augmented_data_generator(img_slice, mask_slice, config):
    
#    orig_img_slice = img_slice.copy()
#    orig_mask_slice = mask_slice.copy()
    
    rotations = config["rotations"] if bool(config["rotation"]) else [None]
    scales = config["scales"] if bool(config["scale"]) else [None]
    shears = config["shears"] if bool(config["shear"]) else [None]
    
    
    for rot in rotations:
        for she in shears:
            for scl in scales:
                yield transform_data_affine(img_slice, mask_slice, 
                                            scale=scl, rotation=rot, shear=she)


def extract_slices(data, mask, config, imgname):
    
    n_slices = data.shape[0]
    for i in xrange(n_slices):
        img_slice, mask_slice = data[i], mask[i]
        
        
        for new_img, new_mask, lab in augmented_data_generator(img_slice,
                                                               mask_slice,
                                                               config):
            #ulozeni extrahovanych dat                                                       
            new_name = imgname + str("%03d" % int(i)) + lab + ".png"
            cv2.imwrite(config["slices_path"] + new_name, new_img)
            cv2.imwrite(config["masks_path"] + new_name, new_mask)

    


def extract_data(imgnames, suffix=".pklz", config={}, to_extract=True):
    """ Ulozi CT rezy do dane slozky """
    
    hs, ws = [], []
    
    for imgname in imgnames:
        print "   Zpracovavam obrazek "+imgname
        data, gt_mask, voxel_size = tools.load_pickle_data(imgname)
        name = imgname[:-len(suffix)]
        # Aplikace CT okenka:
        data = se.apply_CT_window(data)#, target_range=1.0, target_dtype=float)
        
        hs.append(data.shape[1])
        ws.append(data.shape[2])
        
        # extrakce CT oken
        if to_extract:
            extract_slices(data, gt_mask, config, name)
    
    
    avg_shape = [np.mean(hs), np.mean(ws)]
    print "[RESULT] Prumerny tvar rezu je: ", avg_shape
    print "               -> zaokrouhleno: ", [int(avg_shape[0] // 4)*4,
                                               int(avg_shape[1] // 4)*4]

if __name__ =='__main__':
    
    config = read_config()["keras"]
#    # vyprazdneni stareho obsahu slozky
#    clean_folders(config, "folders_to_clean")
#    # zaloha astarych anotaci
#    fh.make_backup(foldername="bounding_boxes", suffix="kerasdata")
    
    suffix = ".pklz"
    imgnames = [imgname for imgname in os.listdir(os.path.dirname(os.path.abspath(__file__))) if imgname.endswith(suffix)]
    
    extract_data(imgnames, suffix=suffix, config=config, to_extract=bool(1))
    
    
    
    
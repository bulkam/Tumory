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


def resize_data(img, mask, new_shape=(316, 352)): # avg=(232, 240); max=(316, 352)
    new_img = cv2.resize(img.astype("uint8"), new_shape[::-1], interpolation = cv2.INTER_CUBIC)
    new_mask = cv2.resize(mask, new_shape[::-1], interpolation = cv2.INTER_NEAREST)
    return new_img, new_mask


def fillin_slice_to_ratio(img, mask, new_shape=(232, 240)):
    """ Doplni zbytek obrazku nulami """
    
    avg_h, avg_w = new_shape
    ratio = float(avg_h) / avg_w

    h, w = img.shape
    
    new_img = np.zeros(img.shape)
    new_mask = np.zeros(mask.shape)
    new_w = w
    new_h = h
    
    if h > w * ratio:
        new_w = np.ceil(float(h) / ratio).astype(int)
        new_h = h
    elif h < w * ratio :
        new_h = np.ceil(w * ratio).astype(int)
        new_w = w
    else:
        new_h = h
        new_w = w
    
    new_img = np.zeros((new_h, new_w))
    new_img[:h, :w] = img
    new_mask = np.zeros((new_h, new_w))
    new_mask[:h, :w] = mask
    
#    print new_img.shape
#    print float(new_h) / new_w
    
    new_img, new_mask = resize_data(new_img, new_mask, new_shape=new_shape)
    
    return new_img, new_mask
    


def transform_data_affine(data, mask, scale=(1, 1), rotation=0, shear=0, 
                          new_shape=(232, 240)):
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
    
    # oriznuti dat
    new_data, new_mask = cut_image(new_data, new_mask)
    # resize dat se zachovanim ratia    
    new_data, new_mask = fillin_slice_to_ratio(new_data, new_mask,
                                               new_shape=new_shape)
    
    return new_data, new_mask, lab


def apply_intensity_augmentation(all_data, sigma):
    """ Prida do obrazku aditivni sum """
    
    data, mask, lab = all_data
    lab = lab + "_noise="+str(int(sigma))
    # ted uz generator sumu
    Na = np.random.normal(loc=0.0, scale=sigma, size=data.shape)
    #Ua = sigma * ( np.random.uniform(size=data.shape) - 0.5 )
    # pricteni aditivniho sumu
    augmented_data = data + Na
    # omezeni na interval
    augmented_data[augmented_data >= 255] = 255
    augmented_data[augmented_data <= 0] = 0
    
    return augmented_data.astype("uint8"), mask, lab
    

def augmented_data_generator(img_slice, mask_slice, config,
                             new_shape=(232, 240)):
    
    rotations = config["rotations"] if bool(config["rotation"]) else [None]
    scales = config["scales"] if bool(config["scale"]) else [None]
    shears = config["shears"] if bool(config["shear"]) else [None]
    intensity_noise_scales = config["intensity_noise_scales"] if bool(config["noise"]) else [0]
    
    for rot in rotations:
        for she in shears:
            for scl in scales:
                affine_transformed_data = transform_data_affine(img_slice, 
                                                               mask_slice, 
                                                               scale=scl, 
                                                               rotation=rot, 
                                                               shear=she,
                                                               new_shape=new_shape)
                for noise_scale in intensity_noise_scales:
                    yield apply_intensity_augmentation(affine_transformed_data,
                                                       sigma=noise_scale)


def extract_slices(data, mask, config, imgname, zero_background=False,
                   new_shape=(232, 240)):
    
    hs, ws = [], []
    
    n_slices = data.shape[0]
    for i in xrange(n_slices):
        img_slice, mask_slice = data[i], mask[i]
        
        
        for new_img, new_mask, lab in augmented_data_generator(img_slice,
                                                               mask_slice,
                                                               config,
                                                               new_shape=new_shape):
            # prekresleni backgroundu na 0    
            if zero_background:            
                new_img[new_mask == 0] = 0
                
            # ulozeni extrahovanych dat 
            new_name = imgname + str("%03d" % int(i)) + lab + ".png"
            cv2.imwrite(config["slices_path"] + new_name, new_img)
            cv2.imwrite(config["masks_path"] + new_name, new_mask)
            
            hs.append(new_img.shape[0])
            ws.append(new_img.shape[1])

    return max(hs), max(ws)


def extract_data(imgnames, suffix=".pklz", config={}, to_extract=True, 
                 zero_background=False, den=32, new_shape=(232, 240)):
    """ Ulozi CT rezy do dane slozky """
    
    hs, ws = [], []
    ratios = []
    mhs, mws = [], []
    
    for imgname in imgnames:
        print "   Zpracovavam obrazek "+imgname
        data, gt_mask, voxel_size = tools.load_pickle_data(imgname)
        name = imgname[:-len(suffix)]
        # Aplikace CT okenka:
        data = se.apply_CT_window(data)#, target_range=1.0, target_dtype=float)
        
        hs.append(data.shape[1])
        ws.append(data.shape[2])
        ratios.append(float(data.shape[1]) / data.shape[2])
              
        # extrakce CT oken
        if to_extract:
            mh, mw = extract_slices(data, gt_mask, config, name, 
                                    zero_background=zero_background,
                                    new_shape=new_shape)
            mhs.append(mh)
            mws.append(mw)

    avg_shape = [np.mean(hs), np.mean(ws)]
    
    round_avg_shape = [int((avg_shape[0] + den/2) // den) * den,
                       int((avg_shape[1] + den/2) // den) * den]
    
    print "[RESULT] Prumerny tvar rezu je: ", avg_shape
    print "               -> zaokrouhleno: ", round_avg_shape
    
    if to_extract:
        max_shape = [np.max(mhs), np.max(mws)]
        round_max_shape = [int((max_shape[0] + den/2) // den) * den,
                           int((max_shape[1] + den/2) // den) * den]
        print "               -> ratio:        ", avg_shape[0] / avg_shape[1]
        print "[RESULT] Maximalni tvar rezu je: ", max_shape
        print "               -> zaokrouhleno: ", round_max_shape
#    plt.hist(ratios, bins=100)
#    plt.show()
    return tuple(round_avg_shape)

if __name__ =='__main__':

    # vytvori potrebne cesty
    fh.create_paths()
    
    config = read_config()["keras"]
    # vyprazdneni stareho obsahu slozky
    clean_folders(config, "folders_to_clean")

    
    suffix = ".pklz"
    imgnames = [imgname for imgname in os.listdir(os.path.dirname(os.path.abspath(__file__))) if imgname.endswith(suffix)]
    
    den = 32, 64, 81
    new_shape = extract_data(imgnames, suffix=suffix, config=config, den=16, 
                             to_extract=bool(0), zero_background=bool(1))
    extract_data(imgnames, suffix=suffix, config=config, 
                 to_extract=bool(1), zero_background=bool(1),
                 new_shape=new_shape)
    
    
    
    

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 13:47:15 2017

@author: mira
"""

from imtools import tools
import matplotlib.pyplot as plt

import skimage
import skimage.segmentation as skiseg
from skimage.morphology import label
import skimage.transform as tf

import SimpleITK as sitk

import numpy as np
import scipy
import os
import json
import copy

import pickle
import cPickle

import slice_executer as se
import file_helper as fh


def clean_boxes(bounding_boxes, min_size=(4, 4)):
    """ Odstrani velmi male bounding boxy """
    
    miny, minx = min_size
    
    for imgname, boxes in bounding_boxes.items():
        for box in boxes:
            (y, h, x, w) = box
            if not (abs(h-y)>=miny and abs(x-w)>=minx):
                boxes.remove(box)
                
    return bounding_boxes


def load_CT_and_make_augmented(imgname, bounding_boxes, suffix='.pklz', 
                               transform_type="similarity", to_show=False,
                               HNM=False, each_to_HNM = 1, config={}):
    """ Nacte CT a ulozi slice a pripadny bounding box """
    
    print "   Zpracovavam obrazek "+imgname
    data, gt_mask, voxel_size = tools.load_pickle_data(imgname)
    n_of_slices = data.shape[0]
    n_negative_slices = 0
    suffix_len = len(suffix)
    transform_type=config["transform"]
    
    # Aplikace CT okenka:
    data = se.apply_CT_window(data)#, target_range=1.0, target_dtype=float)
    
    # Smycka pres kazdy rez
    for i in xrange(n_of_slices):
        print i,
        
        # vytahnuti rezu dat a masky
        data_slice_orig = data[i]
        mask_slice_orig = gt_mask[i]
        
        # pripadne vykresleni
        if to_show:
            se.show_image_in_new_figure(data_slice_orig)
            se.show_image_in_new_figure(mask_slice_orig)
            print np.unique(mask_slice_orig)
        
        # kazdy nekolikaty negativni rez dat do hard_negatives
        if HNM:
            orig_boxes = se.get_bounding_boxes(mask_slice_orig)
            
            if not (len(orig_boxes) >= 1):
                n_negative_slices += 1
                
                if (n_negative_slices % each_to_HNM == 0):
                    img_id = imgname[0:-suffix_len]+str("%03d" % int(i))+suffix
                    #skimage.io.imsave("Negatives/"+imgname[0:-suffix_len]+str("%03d" % int(i))+".png", data_slice_orig)
                    se.save_obj(data_slice_orig.astype("uint8"), "Augmented/Hard_negative_mining/"+img_id)
                    se.save_obj(mask_slice_orig.astype("uint8"), "Masks/"+img_id)
                
        # Augmentace dat
        for data_slice, mask_slice, aug_label in augmented_data_generator(data_slice_orig, mask_slice_orig, transform=transform_type, config=config):
            # pripadne vykreslovani
            if to_show:
                se.show_image_in_new_figure(data_slice)
                se.show_image_in_new_figure(mask_slice)
                print np.unique(mask_slice)
                return False
                
            boxes = se.get_bounding_boxes(mask_slice)
            
            img_id = imgname[0:-suffix_len]+str("%03d" % int(i))+aug_label+suffix
            
            # pokud jsou nejake bounding boxy, mel by je vytahnout a obraz ulozit jako positives
            if len(boxes) >= 1:
                # Ulozeni bounding boxu
                bounding_boxes["datasets/processed/orig_images/"+img_id] = boxes
                # Ulozeni obrazku
                #skimage.io.imsave("Slices/"+imgname[0:-suffix_len]+str("%03d" % int(i))+".png", data_slice)
                se.save_obj(data_slice, "Augmented/Slices/"+img_id)
            
            # jinak je da mezi negatives     
            else:
                # Ulozeni obrazku
                frame_id = 0
                for data_to_save in se.liver_inside_only_generator(data_slice, mask_slice, 
                                                                   fill_holes=False, 
                                                                   metric='euclides'): # u normalnich davam taxicab
                    if not data_to_save is None:
                        #skimage.io.imsave("Negatives/"+imgname[0:-suffix_len]+str("%03d" % int(i))+".png", data_to_save)
                        se.save_obj(data_to_save, "Augmented/Negatives/"+imgname[0:-suffix_len]+str("%03d" % int(i))+"#"+str("%03d" % int(frame_id))+aug_label+suffix)
                        frame_id += 1
            
    print ""
   

def remove_small_objects(mask, minval=32):
    """ Odstrani napriklad male sede na okrajich """
    
    max_val = np.max(mask)
    min_val = np.min(mask)
    
    non_extrema = (mask > min_val) & (mask < max_val)
    non_small = skimage.morphology.remove_small_objects(non_extrema, minval)
    
    mask[(non_extrema>=1) & (non_small==0)] = max_val
    
    return mask


def transform_mask(mask, g_transform):
    """ Transformuje masku tak, aby nedoslo k zadne neurcitosti """
    
    # prazdny obrazek, kam se bude doplnovat
    blank = np.zeros(mask.shape)
    # maska jater i s artefakty
    liver = (mask >= 1)
    new_liver = tf.warp(liver, g_transform)
    # maska pouze artefaktu
    artefacts = (mask == 1)
    new_artefacts = tf.warp(artefacts, g_transform)
    # vykresleni do vysledneho snimku
    blank[new_liver >= 0.5] = 2
    blank[new_artefacts >= 0.5] = 1

    return blank


# TODO: jeste mozna zkusit scale
# ty artefakty se dost casto oriznou
def transform_data_sim(data, mask, rotation=0, scale=1, intensity_factor=0):
    """ Augmentace dat - rotovana a zvetsena data """
    
    lab = "rot="+str(rotation)+"_int="+str(intensity_factor)
    
    # padding obrazku, aby byla rezerva a neorezavalo se to
    new_data = np.zeros((data.shape[0]*2, data.shape[1]*2))
    new_data[new_data.shape[0]//4: 3*new_data.shape[0]//4, new_data.shape[1]//4: 3*new_data.shape[1]//4] = copy.copy(data)
    new_mask = np.zeros((data.shape[0]*2, data.shape[1]*2))
    new_mask[new_mask.shape[0]//4: 3*new_mask.shape[0]//4, new_mask.shape[1]//4: 3*new_mask.shape[1]//4] = copy.copy(mask)    
    
    dh, dw = np.array(new_data.shape) / 2
    # zakladni transformace
    basic = tf.SimilarityTransform(scale=scale, rotation=np.deg2rad(rotation))
    # korekce posunuti
    trans = tf.SimilarityTransform(translation=[-dw, -dh])
    transinv = tf.SimilarityTransform(translation=[dw, dh])
    # vysledna transformace
    g_transform = trans + (basic + transinv)
    # aplikace transformace
    new_data = tf.warp(new_data.astype("float64"), g_transform).astype("int")
    new_mask = transform_mask(new_mask, g_transform).astype("int")
    # korekce
    #se.show_image_in_new_figure(new_mask)
    #new_mask = scipy.ndimage.filters.maximum_filter(new_mask, size=[3, 3])#.astype("int")
    #new_mask = remove_small_objects(new_mask)
    #se.show_image_in_new_figure(new_mask)
    
    return new_data, new_mask, lab


def transform_data_affine(data, mask, scale=(1, 1), rotation=0, 
                         shear=0, intensity_factor=0):
    """ Augmentace dat - affinni transformace """
    
    lab = "rot="+str(rotation)+"_shear="+str(int(shear*10))
    lab = lab + "_scale="+str(int(scale[0]*10))+"x"+str(int(scale[1]*10))
    lab = lab + "_int="+str(intensity_factor)
    
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
    new_data = tf.warp(new_data.astype("float64"), g_transform).astype("int")
    #new_mask = (tf.warp(new_mask.astype("float64"), g_transform)+0.5).astype("int")
    new_mask = transform_mask(new_mask, g_transform).astype("int")
    # korekce
    #new_mask = remove_small_objects(new_mask)
    
    return new_data, new_mask, lab


def similarity_transform_generator(data, mask, config={}):
    """ Generator transformovanych dat """
    
    rotations = config["rotations"]
    
    for rotation in rotations:
        yield transform_data_sim(data, mask, rotation=rotation)


def affine_transform_generator(data, mask, config={}):
    """ Generator transformovanych dat - affinni """  
 
    to_rotate = bool(config["rotation"])
    to_shear = bool(config["shear"])
    
    rotations = config["rotations"] if to_rotate else [0]
    shears = config["shears"] if to_shear else [0]
    
    for rotation in rotations:
        for shear in shears:
            yield transform_data_affine(data, mask, rotation=rotation, shear=shear)
    

def augmented_data_generator(data, mask, transform="similarity", config={}):
    """ Generator augmentovanych dat """
    
    if transform == "similarity":
        # generator similarity transformovanych dat
        for data, mask, lab in similarity_transform_generator(data, mask, config=config):
            aug_label = "SIM"+lab
            yield data, mask, aug_label
    
    elif transform == "affine":
        # generator affine transformovanych dat
        for data, mask, lab in affine_transform_generator(data, mask, config=config):
            aug_label = "AFFINE"+lab
            yield data, mask, aug_label
            

def main():
    print "--- Augmentace dat ---"
    
    config = se.read_config()
    # vyprazdneni stareho obsahu slozky
    se.clean_folders(config, "augmented_folders_to_clean")
    # zaloha starych anotaci
    fh.make_backup(foldername="bounding_boxes", suffix="augmentation")
    
    imgnames = [imgname for imgname in os.listdir(os.path.dirname(os.path.abspath(__file__))) if imgname.endswith('.pklz')]
    #print imgnames
    
    bounding_boxes = dict()  
    
    for imgname in imgnames:
        # nacteni dat
        load_CT_and_make_augmented(imgname, bounding_boxes, to_show=(1==0), config=config)
        #break
     
    # vymazani velmi malych bounding boxu
    #bounding_boxes = clean_boxes(bounding_boxes)
     
    # pridani bounding boxu bez augmentace
    bounding_boxes_orig = se.precti_json("bounding_boxes/bounding_boxes.json")
    bounding_boxes.update(bounding_boxes_orig)
    # ulozani vsech anotaci
    se.zapis_json(bounding_boxes, "bounding_boxes/bb_augmented.json")


if __name__ =='__main__':
    se.main()   # klasicka extrakce dat
    main()      # augmentace dat

    
    
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:07:09 2017

@author: mira
"""

import matplotlib.pyplot as plt
import skimage

import numpy as np
import os
import re

import slice_executer as se


def show_frames_in_image(img, boxes):
    """ Vykresli obrazek a do nej prislusne framy """
    plt.figure()
    skimage.io.imshow(img, cmap = "gray")
    
    for box in boxes:
        x, h, y, w = box

        plt.plot([y, w], [x, x], "r", lw = "3")
        plt.plot([y, y], [x, h], "r", lw = "3")
        plt.plot([w, w], [x, h], "r", lw = "3")
        plt.plot([y, w], [h, h], "r", lw = "3")

    plt.show()


def draw_all_negatives(ref_folder="/Augmented/Negatives",
                       target_folder="Annotations_testing/Augmented/Negatives/"):
    """ Vykresli vsechny negativni obrazky a ulozi jako png """
    
    imgnames = [imgname for imgname in os.listdir(os.path.dirname(os.path.abspath(__file__))+ref_folder) if imgname.endswith('.pklz')]
    
    for imgname in imgnames:
        img = se.load_obj(ref_folder+"/"+imgname)
        skimage.io.imsave(target_folder+imgname[0:-5]+".png", img)


def test_it_all(imgnames, annotations, folder="/Augmented/Slices", 
                target_folder="Annotations_testing/Slices/"):
    """ Ulozi vsechny data i s labelem jako png obrazek """
    
    i = 0
    for imgname in imgnames:
        i+=1
        print i,
        fname = "datasets/processed/orig_images/"+imgname
        img = se.load_obj(folder+"/"+imgname)
        boxes = annotations[fname]
        
        plt.figure()
        plt.ioff()
        plt.imshow(img, cmap = "gray")
        
        for box in boxes:
            x, h, y, w = box
    
            plt.plot([y, w], [x, x], "r", lw = "3")
            plt.plot([y, y], [x, h], "r", lw = "3")
            plt.plot([w, w], [x, h], "r", lw = "3")
            plt.plot([y, w], [h, h], "r", lw = "3")

        plt.savefig(target_folder+imgname[0:-5])
        plt.close()


def test_boxes_compatibility():
    """ Zkotroluje, jak moc se augmentovane boxy lisi od originalu """
    
    orig = se.precti_json("bounding_boxes/bounding_boxes.json")
    aug = se.precti_json("bounding_boxes/bb_augmented.json")
    
    suffix = ".pklz"
    for key in orig.keys():
        img_id = key[:-len(suffix)]
        related = [name for name in aug.keys() if img_id in name]
        for imgname in related:
            orig_boxes = orig[key]
            aug_boxes = aug[imgname]
            if not len(orig_boxes) == len(aug_boxes):
                print imgname
                print orig_boxes," != ",aug_boxes
                print "----------------------"


def test_annotations(mode_aug=0, to_show=False):
    """ Provede test anotaci """
    
    print "[INFO] Testuji anotace ",
    print "dat bez augmentace." if mode_aug==0 else "augmentovanych dat." 
    # 0: augmentovana, jinak: originalni
    orig_folder = "/Slices"
    aug_folder = "/Augmented/Slices"
    
    # tady menit
    folder = orig_folder if mode_aug==0 else aug_folder
    
    imgnames = [imgname for imgname in os.listdir(os.path.dirname(os.path.abspath(__file__))+folder)  if imgname.endswith('.pklz')]
    print "[INFO] Celkem obrazku: ", len(imgnames)
    
    annotations = se.precti_json("bounding_boxes/bounding_boxes.json") if mode_aug==0 else se.precti_json("bounding_boxes/bb_augmented.json")
    annotations_keys = annotations.keys()
    
    # specifikace puvodnich a cilovych adresaru
    ref_folder_slices = "/Slices" if mode_aug==0 else "/Augmented/Slices"
    ref_folder_negatives = "/Negatives" if mode_aug==0 else "/Augmented/Negatives"
    target_folder_slices = "Annotations_testing/Slices/" if mode_aug==0 else "Annotations_testing/Augmented/Slices/"
    target_folder_negatives = "Annotations_testing/Negatives/" if mode_aug==0 else "Annotations_testing/Augmented/Negatives/"
    # testovani vsech augmentovanych dat
    test_it_all(imgnames, annotations, folder=ref_folder_slices, target_folder=target_folder_slices)
    draw_all_negatives(ref_folder=ref_folder_negatives, target_folder=target_folder_negatives)
    
    # nahodne zobrazeni nejakych
    if to_show:
        indexes = (np.random.rand(10)*len(annotations_keys)).astype(int)
        for i in indexes:
            print i, ": ",
            
            fname = re.sub("datasets/processed/orig_images", folder[1:], annotations_keys[i])
            print fname
            
            boxes = annotations[annotations_keys[i]] 
            img = se.load_obj(fname)
            
            show_frames_in_image(img, boxes)
            
    print "Hotovo !"
        

if __name__ =='__main__':
    
    config = se.read_config()
    
    """
    se.clean_folders(config, "annotations_testing_folders_to_clean")
    test_annotations(mode_aug=0)
    
    se.clean_folders(config, "annotations_testing_augmented_folders_to_clean")
    test_annotations(mode_aug=1)"""
    
    test_boxes_compatibility()

    
    
    

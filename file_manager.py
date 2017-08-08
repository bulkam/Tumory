# -*- coding: utf-8 -*-
"""
Created on Mon Aug 07 09:48:07 2017

@author: mira
"""

import os
from os import path as op
from shutil import copyfile
from shutil import move as movefile
from shutil import copytree

import json
import copy
import glob

import pickle
import cPickle
import time
import re

import data_reader as dr


class Manager:
    def __init__(self):
        
        self.dataset = dr.DATAset()
        self.config = self.dataset.config
        
    
    def clean_folder(self, foldername):
        """ Vymaze vsechny soubory ve slozce """
        
        foldername = os.path.dirname(os.path.abspath(__file__))+"/"+foldername+'*'
        files = glob.glob(foldername)
        
        for f in files:
            os.remove(f)
        
        print "[INFO] Veskery obsah slozky "+str(foldername)+" byl vymazan."
    
    
    def clean_folders(self, config, section_name):
        """ Pro vsechny slozky ze seznamu vymaze jejich obsah """
        
        for folder in config[section_name]:
            self.clean_folder(config[folder])


    def make_folder(self, foldername):
        """ Vytvori novou slozku, pokud jeste neexistuje """
        
        # vytvoreni cesty
        newpath = str(os.path.dirname(os.path.abspath(__file__)))+'/'+foldername
        
        # vytvorit slozku, pokud neexistuje
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        
    def make_dataset_backup(self, path="bounding_boxes", suffix="",
                            prefix="Z-OLD", mode="copy"):
        """ Cely obseh slozky zkopiruje do podslozky old """
    
        print "[INFO] Zalohuji obrazky "
    
        new_foldername = self.get_next_subfolder_id(path)
        # zjisteni vsech podslozek
        subfolders = [f for f in os.listdir(path) if (op.isdir(op.join(path, f)) and not f.startswith(prefix))]
        
        for foldername in subfolders:
            print "   - ", foldername
        
            # vytvori slozku Z-OLD s podslozkou pro aktualni kopii, pokud neexistuje          
            newpath = path+"/"+new_foldername+"/"+foldername
            
            if mode=="move":
                # vytvoreni slozky
                if not op.exists(newpath):
                    os.makedirs(newpath)
                
                # kopirovani vsech souboru podslozky
                foldername = path+"/"+foldername
                for f in os.listdir(foldername):
                    if op.isfile(op.join(foldername, f)):
                        if mode=="move" and not f.startswith("00_copy"):
                            movefile(foldername+"/"+f, newpath+"/"+f)
                        else:
                            copyfile(foldername+"/"+f, newpath+"/"+f)
                            
            else:
                foldername = path+"/"+foldername
                copytree(foldername, newpath+"/")

        # bounding boxy
        print "[INFO] Zalohuji anotace (bounding boxy)"
        
        boxes_path = op.dirname(op.dirname(op.abspath(__file__))+'/'+self.config["annotations_path"])
        new_foldername = self.get_next_subfolder_id(boxes_path)
        
        # vytvori slozku Z-OLD s podslozkou pro aktualni kopii, pokud neexistuje
        newpath = boxes_path+"/"+new_foldername
        
        # vytvoreni slozky
        if not op.exists(newpath):
            os.makedirs(newpath)
            
        # kopirovani souboru
        for f in os.listdir(boxes_path):
            if op.isfile(op.join(boxes_path, f)):
                if mode=="move":
                    movefile(boxes_path+"/"+f, newpath+"/"+f)
                else:
                    copyfile(boxes_path+"/"+f, newpath+"/"+f)           

    
    def get_next_subfolder_id(self, path="", min_id=1000, prefix="Z-OLD"):
        """ Vrati, jaky bude pristi nazev slozky se zalohou """
        
        # nalezeni podslozek
        #path = self.config["test_images_path"]
        subfolders = [f for f in os.listdir(path) if (os.path.isdir(op.join(path, f)) and f.startswith(prefix))]
        # prochazeni nazvu podslozek
        for f in subfolders:
            try:
                min_id = min(min_id, int(re.findall(r'\d{3}', f)[0]))
            except:
                pass
        
        return prefix + str(min_id-1)
    
    
    def get_imgname(self, path):
        """ Vrati jmeno obrazku bez pripony a cesty """
        
        dot = re.findall('[^\/]*\.', path)
        mesh = re.findall('[^\/]*\#', path)
        
        return dot[0][:-1] if len(mesh)==0 else mesh[0][:-1]
        

    def choose_test_images(self, positives_path, negatives_path, n_to_test=10):
        """ Vybere nekolik obrazku a oznaic je za testovaci """
        
        positives = os.listdir(positives_path)
        negatives = os.listdir(negatives_path)
        each_to_test_p = 2 * len(positives) // n_to_test
        each_to_test_n = 2 * len(negatives) // n_to_test
        
        each_to_tests = [each_to_test_p, each_to_test_n]
        sources = [positives_path, negatives_path]
        
        test_images = list()
        
        for s, source_folder in enumerate(sources):
            
            each_to_test = each_to_tests[s]
            i = 0
            
            for f in os.listdir(source_folder):
                if op.isfile(op.join(source_folder, f)):
                    i += 1
                    if i % each_to_test == 0:                        
                        test_images.append(self.get_imgname(f))
                        
        return test_images
                    

    def update_dataset(self, configpath="CTs/Configuration/config.json", 
                       mode_aug=False, source="CTs/", n_to_test=10,
                       mode="copy"):
        """ Aktualizuje dataset v hlavni slozce a stary zalohuje """

        # zaloha dat
        path = op.abspath(op.join(self.config["test_images_path"], os.pardir))
        self.make_dataset_backup(path=path, mode=mode)
        
        data_extraction_config = self.dataset.precti_json(configpath)
        # originalni data
        positives_path = source + data_extraction_config["positives_path"]
        negatives_path = source + data_extraction_config["negatives_path"]
        HNM_path = source + data_extraction_config["HNM_path"]
        masks_path = source + data_extraction_config["masks_path"]
        bb_path = source + data_extraction_config["bounding_boxes_path"]
        # augmentovana data
        augmented_positives_path = source + data_extraction_config["augmented_positives_path"]
        augmented_negatives_path = source + data_extraction_config["augmented_negatives_path"]
        augmented_HNM_path = source + data_extraction_config["augmented_HNM_path"]
        
        from_to = {positives_path: self.dataset.orig_images_path,
                   negatives_path: self.dataset.negatives_path,
                   HNM_path: self.dataset.HNM_images_path,
                   masks_path: self.dataset.masks_path,
                   bb_path: self.dataset.annotations_path}
        # pripadne pridani augmentovanych dat
        from_to_augmented = {augmented_positives_path: self.dataset.orig_images_path,
                             augmented_negatives_path: self.dataset.negatives_path,
                             augmented_HNM_path: self.dataset.HNM_images_path}
        
        # vyber testovacich obrazku
        test_images = self.choose_test_images(positives_path, negatives_path)
        target_folder_test = self.dataset.test_images_path
        # augmentovana az nakonec
        items = from_to.items()+from_to_augmented.items()

        for item in  items:

            source_folder = item[0]
            target_folder = item[1]
            
            print "[INFO] Kopiruji soubory "
            print "    z ", source_folder
            print "   do ", target_folder
            
            # anotace - bounding boxy
            if source_folder == bb_path:
                if mode_aug:
                    copyfile(source_folder+"/"+"bb_augmented.json", target_folder) 
                else:
                    copyfile(source_folder+"/"+"bounding_boxes.json", target_folder) 
            # jinak jde o obrazky
            else:
                for f in os.listdir(source_folder):
                    if op.isfile(op.join(source_folder, f)):
                        
                        if source_folder in (positives_path, 
                                             negatives_path, 
                                             HNM_path):
        
                            if self.get_imgname(f) in test_images:
                                print "[INFO] Testovaci obrazek: ",f
                                # negativni jsou jen vyrezy
                                #  -> dame tam cele rezy z HNM slozky
                                if not source_folder == negatives_path:
                                    copyfile(source_folder+"/"+f, target_folder_test+"/"+f)
                                continue
                        
                        elif source_folder in (augmented_positives_path, 
                                               augmented_negatives_path, 
                                               augmented_HNM_path):
                            
                            found_in_test_images = False
                            
                            for imgname in test_images:
                                if f.startswith(imgname):
                                    print "[WARNING] "+f+" je testovaci obrazek -> nekopirovat!"
                                    found_in_test_images = True
                                    break
                                
                            if found_in_test_images:
                                continue
                    
                    # kopirovani souboru
                    copyfile(source_folder+"/"+f, target_folder+"/"+f) 


if __name__ =='__main__':
    # inicializace
    manager = Manager()
    # aktualizace datasetu
    manager.update_dataset(mode_aug=True, mode="move")

    


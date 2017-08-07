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
                        if mode=="move":
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
                

    def update_dataset(self, configpath="CTs/Configuration/config.json"):
        """ Aktualizuje dataset v hlavni slozce a stary zalohuje """

        # zaloha dat
        path = op.abspath(op.join(self.config["test_images_path"], os.pardir))
        self.make_dataset_backup(path=path)
        
        data_extraction_config = self.dataset.precti_json(configpath)


if __name__ =='__main__':
    # inicializace
    manager = Manager()
    # aktualizace datasetu
    manager.update_dataset()

    


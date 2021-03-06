# -*- coding: utf-8 -*-
"""
Created on Sun Aug 06 21:58:25 2017

@author: mira
"""

import os
from os import path as op
from shutil import copyfile
import json
import copy
import glob

import pickle
import cPickle
import time


def load_json(name):
    """ Nacte .json soubor a vrati slovnik """
    filepath = os.path.dirname(os.path.abspath(__file__))+"/"+str(name)
    mydata = {}
    with open(filepath) as d:
        mydata = json.load(d)
        d.close()
    return mydata


def make_folder(foldername):
    
    # vytvoreni cesty
    newpath = str(os.path.dirname(os.path.abspath(__file__)))+'/'+foldername
    
    # vytvorit slozku, pokud neexistuje
    if not os.path.exists(newpath):
        os.makedirs(newpath)


def make_backup(foldername="bounding_boxes", suffix=""):
    """ Cely obseh slozky presune do podslozky old """
    
    # vytvori slozku old s podslozkou pro aktualni kopii, pokud neexistuje
    timestamp = str(int(time.time()*100))
    newpath = foldername+"/old/"+timestamp+suffix
    make_folder(newpath)
    
    # kopirovani vsech souboru podslozky
    for f in os.listdir(foldername):
        if op.isfile(op.join(foldername, f)):
            copyfile(foldername+"/"+f, newpath+"/"+f)


def create_paths():
    """ Vytvori potrebne adresare """
    
    folders = ["Augmented/Hard_negative_mining/",
               "Augmented/Slices/",
               "Augmented/Negatives",
               "Augmented/Masks/",
               "Slices/",
               "Negatives/",
               "Hard_negative_mining",
               "Masks/",
               "bounding_boxes",
               "kerasdata/Slices/",
               "kerasdata/Masks/",
               "frames/hnm/"]
    
    for path in folders:
        make_folder(path)

    config = load_json("Configuration/config.json")

    for key, path in config.items():
        if "path" in key and not "." in path:
            make_folder(path)

    

if __name__ =='__main__':
    # vytvoreni zalohy
    make_backup(suffix="testing")
    # vytvoreni slozek
    create_paths()

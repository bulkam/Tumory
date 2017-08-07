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
            

if __name__ =='__main__':
    make_backup(suffix="testing")

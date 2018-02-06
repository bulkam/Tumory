# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 15:52:12 2018

@author: Mirab
"""

import os
import json


def make_folder(foldername):
    """ Vytvori novou slozku, pokud jeste neexistuje """
    
    # vytvoreni cesty
    newpath = str(os.path.dirname(os.path.abspath(__file__)))+'/'+foldername
    
    # vytvorit slozku, pokud neexistuje
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        
               
def save_json(jsondata,  name):
    """ Ulozi slovnik do .json souboru """
    filepath = os.path.dirname(os.path.abspath(__file__))+"/"+str(name)
    with open(filepath, 'w') as f:
        json.dump(jsondata, f)
        

def load_json(name):
    """ Nacte .json soubor a vrati slovnik """
    filepath = os.path.dirname(os.path.abspath(__file__))+"/"+str(name)
    mydata = {}
    with open(filepath) as d:
        mydata = json.load(d)
        d.close()
    return mydata
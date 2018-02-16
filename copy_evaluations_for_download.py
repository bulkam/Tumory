# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 00:18:12 2018

@author: Mirab
"""

import glob
import os
import re
from shutil import copyfile

import file_manager_metacentrum as fm


foldername = "experiments/aug_structured_data-liver_only/*"
folders = glob.glob(foldername)

for folder in folders:
    
    name = re.sub(r".+\\+", "", folder)
    if "." in name:
        continue
    new_name = "experiments_results_copy/"
    fm.make_folder(new_name + name + "/logs")
    
    files = os.listdir(folder)
    for f in files:
        if f.endswith(".json"):
            copyfile(folder + "/" + f, 
                     new_name + name + "/" + f)
        if f.endswith("logs"):
            logfiles = os.listdir(folder + "/" + f)
            for lf in logfiles:
                if not (lf.endswith(".hdf5") or lf.endswith(".h5")):
                    copyfile(folder + "/" + f + "/" + lf, 
                             new_name + name + "/" + f + "/" + lf)            
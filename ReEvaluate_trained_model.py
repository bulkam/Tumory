# -*- coding: utf-8 -*-
"""
Created on Mon Feb 05 13:18:56 2018

@author: mira
"""

print("[INFO] START")


import keras.backend as K
from keras.models import load_model

import sys
import h5py

#import keras_data_reader as dr
import file_manager_metacentrum as fm
import CNN_experiment

print("[INFO] Vse uspesne importovano - OK")


experiment_foldername = str(sys.argv[1])
hogs_only = False

if len(sys.argv) >= 3:
    if "hog" in str(sys.argv[2]).lower():
        hogs_only = True

""" Nacteni dat """

#experiment_name = "no_aug_structured_data-liver_only"
experiment_name = "aug_structured_data-liver_only"
#experiment_name = "aug-ge+int_structured_data-liver_only"

hdf_filename = "datasets/processed/"+experiment_name+".hdf5"
hdf_file = h5py.File(hdf_filename, 'r')


""" Nacteni natrenovaneho modelu """

model = load_model(experiment_foldername+"/model.hdf5")
optimizer = model.optimizer

""" Ulozeni konfigurace """

epochs = 5
class_weight = [0.1, 35.0, 4.0]

config = {"epochs": epochs,
         "class_weight": class_weight,
         "experiment_name": experiment_name,
         "experiment_foldername": experiment_foldername,
         "LR": float(K.eval(optimizer.lr)),
         "optimizer": str(optimizer),
         "loss": str(model.loss),
         "metrics": model.metrics}
fm.save_json(config, experiment_foldername+"/notebook_config.json")


""" Ohodnoceni """

if not hogs_only:
    CNN_experiment.evaluate_all(hdf_file, model, experiment_foldername)
else:
    CNN_experiment.evaluate_hogs_only(experiment_foldername)

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 21:52:11 2018

@author: Mirab
"""

import CNN_evaluator
import h5py
import numpy as np
#from matplotlib import pyplot as plt
import file_manager_metacentrum as fm
#import sys


foldername = "experiments/aug_structured_data-liver_only/RMS_SegNet4_LRdet_5epochs_weighted-01-35-4"
#foldername = "classification/Keras/experiments/aug_structured_data-liver_only/RMS_SegNet4_LRdet_5epochs_weighted-01-35-4"

file = h5py.File(foldername + "/test_results.hdf5", 'r')

test_data = file["test_data"]
test_labels = file["test_labels"]
test_predictions = file["test_predictions"]

for key in file.keys():
    print(key)


""" Specifikace testovanych parametru """

min_sizes = np.arange(36)**2
kernel_sizes = np.arange(1, 10)*2 + 1

print("[INFO] ", min_sizes.shape[0] * kernel_sizes.shape[0],
      "parametru se bude testovat")
A = np.zeros((min_sizes.shape[0], kernel_sizes.shape[0]))


""" Evaluace pro ruzne phodnoty parametru """

results_to_save = {}

for i, size in enumerate(min_sizes):
    results_to_save[str(size)] = {}
    for j, ksize in enumerate(kernel_sizes):      
        config = {"min_object_size": size, "element_closing_size": ksize}
        JS = CNN_evaluator.evaluate_JS(test_labels, test_predictions, 
                                       print_steps=bool(0), J_thr=0.8, 
                                       from_config=True, config=config)
        results_to_save[str(size)][str(ksize)] = JS
        A[i, j] = JS["TPR"]
        
best_i, best_j = np.unravel_index(np.argmax(A), A.shape)
best_size, best_ksize = min_sizes[best_i], kernel_sizes[best_j]
print("[RESULT] Nejlepsi TPR ", "%.4f" % A[best_i, best_j], " vykazovalo nastaveni:")
print("         - min_object_size = ", best_size)
print("         - structuring_element_size = ", best_ksize)
print("     Maximalni TPR je ", np.max(A))


""" Ulozeni vysledku """

recalls_to_save = {"min_object_sizes": min_sizes.tolist(),
                   "kernel_sizes": kernel_sizes.tolist(),
                   "A": A.tolist()}
results_to_save.update({"matrix":recalls_to_save})
fm.save_json(results_to_save, foldername + "/morphology_testing.json")

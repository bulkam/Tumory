# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 13:09:14 2017

@author: Mirab
"""

import keras
import h5py
import numpy as np
from random import shuffle
import glob
import skimage.io
import re

from matplotlib import pyplot as plt


def show_results(label, orig, result, lesion, path="classification/Keras/results/comparation", index=0):
    
    f, [[ax11, ax12], [ax21, ax22]] = plt.subplots(2, 2)
    f.set_figheight(9)
    f.set_figwidth(9)

    ax11.imshow(label)
    ax11.set_title('Anotation')
    ax12.imshow(orig, cmap="gray")
    ax12.set_title('Orig image')
    ax21.imshow(result)
    ax21.set_title('Result label')
    ax22.imshow(lesion, cmap="gray")
    ax22.set_title('Predicted lesion')
    #plt.show()
    
    
    plt.savefig(path+"/result_"+str("%.5d" % index)+".png")
    plt.close()


path = "classification/Keras/results/test_results-aug_5epoch_structured_data-liver_only"
path = "classification/Keras/results/test_results-no_aug_20epoch_structured_data-liver_only"
file = h5py.File(path+".hdf5", 'r')

test_data = file["test_data"]
test_labels = file["test_labels"]
test_predictions = file["test_predictions"]

for index in range(test_data.shape[0]):
    if index % 100 == 0:
        print(index)
    
    orig = test_data[index][:, :, 0].astype("uint8")
    label = test_labels[index].astype("uint8") * 255
    result = test_predictions[index].astype("float") 
    lesion = np.argmax(result, axis=2)*127
    
    show_results(label, orig, result, lesion, path=path, index=index)

file.close()
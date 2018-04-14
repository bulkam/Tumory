# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 22:37:44 2018

@author: mira
"""

import keras
import h5py
import numpy as np
from random import shuffle
import glob
import skimage.io
import re

def get_imagename(path):
    """ Vrati jmeno obrazku bez pripony a cesty """
    dot = re.findall('[^\/]*\.', path)
    mesh = re.findall('[^\/]*\#', path)
    return dot[0][:-1] if len(mesh)==0 else mesh[0][:-1]

def get_unique_name(path):
    """ Vrati jmeno obrazku bez pripony a cesty """
    mesh = re.findall(r'.*GT\d{3}', path)[0]
    return mesh

def get_maskname(imgname):
    """ Vrati jmeno obrazku bez pripony a cesty """
    mesh = re.sub(r'Slices', 'Masks', imgname)
    return mesh

def merge_imgnames(addrs, addrs_un):

    k = 0

    train_names = list()
    val_names = list()
    test_names = list()

    for i, name in enumerate(addrs_un):
        k += 1
        if k <= 3:
            train_names.append(name)
        elif k == 4:
            val_names.append(name)
        elif k == 5:
            test_names.append(name)
            k=0
    
    print(len(train_names), len(val_names), len(test_names))
    
    train_addrs = list()
    val_addrs = list()
    test_addrs = list()
    
    for addr in addrs:
        for name in addrs_un:
            if addr.startswith(name):
                if name in train_names:
                    train_addrs.append(addr)
                elif name in val_names:
                    val_addrs.append(addr)
                elif name in test_names:
                    test_addrs.append(addr)
                    
    print(len(train_addrs), len(val_addrs), len(test_addrs))
        
    return train_addrs, val_addrs, test_addrs
    

""" Definice cest """

hdf_path = "datasets/processed/aug_structured_data-liver_only.hdf5"
data_path = 'CTs/kerasdata/Slices/*.png'
masks_path = 'CTs/kerasdata/Masks/*.png'

data_addrs = glob.glob(data_path)
masks_addrs = glob.glob(masks_path)

# bez augmentace
data_addrs = [a for a in data_addrs]# if "rot=0_shear=0" in a]
masks_addrs = [a for a in masks_addrs]# if "rot=0_shear=0" in a]

print(len(data_addrs), len(masks_addrs))


""" Rozdeleni na train / valid / test """
data_addrs_un = list(set([get_unique_name(a) for a in data_addrs]))
masks_addrs_un = [re.sub("Slices", "Masks", a) for a in data_addrs_un]

# Rozdeleni do datasetu podle rezu
train_data_addrs, val_data_addrs, test_data_addrs = merge_imgnames(data_addrs, data_addrs_un)
train_label_addrs, val_label_addrs, test_label_addrs = merge_imgnames(masks_addrs, masks_addrs_un)


""" Shufflovani """
# trenovaci
a = list(zip(train_data_addrs, train_label_addrs))
shuffle(a)
train_data_addrs, train_label_addrs = zip(*a)
# validacni
b = list(zip(val_data_addrs, val_label_addrs))
shuffle(b)
val_data_addrs, val_label_addrs = zip(*b)
# testovaci
c = list(zip(test_data_addrs, test_label_addrs))
shuffle(c)
test_data_addrs, test_label_addrs = zip(*c)


""" Vytvoreni datasetu """

train_shape = (len(train_data_addrs), 240, 232, 1)
val_shape = (len(val_data_addrs), 240, 232, 1)
test_shape = (len(test_data_addrs), 240, 232, 1)

n_classes = 3
train_label_shape = (len(train_data_addrs), 240, 232, n_classes)
val_label_shape = (len(val_data_addrs), 240, 232, n_classes)
test_label_shape = (len(test_data_addrs), 240, 232, n_classes)

print(train_shape, val_shape, test_shape)

# otevreni souboru
hdf5_file = h5py.File(hdf_path, mode='w')

# data
hdf5_file.create_dataset("train_data", train_shape, np.uint8)
hdf5_file.create_dataset("val_data", val_shape, np.uint8)
hdf5_file.create_dataset("test_data", test_shape, np.uint8)

# anotace
hdf5_file.create_dataset("train_labels", train_label_shape, np.int8)
hdf5_file.create_dataset("val_labels", val_label_shape, np.int8)
hdf5_file.create_dataset("test_labels", test_label_shape, np.int8)


""" Ulozeni dat do datasetu """

for i in range(len(train_data_addrs)):
    # progress
    if i % 1000 == 0 and i > 1:
        print('Trenovaci data: {}/{}'.format(i, len(train_data_addrs)))
    # CT rez
    addr = train_data_addrs[i]
    img = skimage.io.imread(addr)
    input_img = np.zeros((img.shape[0], img.shape[1], 1))
    input_img[:, : ,0] = img
    hdf5_file["train_data"][i, ...] = input_img
    # anotace
    addr = train_label_addrs[i]
    mask = skimage.io.imread(addr)
    input_mask = np.zeros((mask.shape[0], mask.shape[1], n_classes))
    for j in range(n_classes):
        input_mask[:, : ,j][mask==j] = 1
    hdf5_file["train_labels"][i, ...] = input_mask
    
for i in range(len(val_data_addrs)):
    # progress
    if i % 1000 == 0 and i > 1:
        print('Validacni data: {}/{}'.format(i, len(val_data_addrs)))
    # CT rez
    addr = val_data_addrs[i]
    img = skimage.io.imread(addr)
    input_img = np.zeros((img.shape[0], img.shape[1], 1))
    input_img[:, : ,0] = img
    hdf5_file["val_data"][i, ...] = input_img
    # anotace
    addr = val_label_addrs[i]
    mask = skimage.io.imread(addr)
    mask = skimage.io.imread(addr)
    input_mask = np.zeros((mask.shape[0], mask.shape[1], n_classes))
    for j in range(n_classes):
        input_mask[:, : ,j][mask==j] = 1
    hdf5_file["val_labels"][i, ...] = input_mask
    
for i in range(len(test_data_addrs)):
    # progress
    if i % 1000 == 0 and i > 1:
        print('Testovaci data: {}/{}'.format(i, len(test_data_addrs)))
    # CT rez
    addr = test_data_addrs[i]
    img = skimage.io.imread(addr)
    input_img = np.zeros((img.shape[0], img.shape[1], 1))
    input_img[:, : ,0] = img
    hdf5_file["test_data"][i, ...] = input_img
    # anotace
    addr = test_label_addrs[i]
    mask = skimage.io.imread(addr)
    mask = skimage.io.imread(addr)
    input_mask = np.zeros((mask.shape[0], mask.shape[1], n_classes))
    for j in range(n_classes):
        input_mask[:, : ,j][mask==j] = 1
    hdf5_file["test_labels"][i, ...] = input_mask

hdf5_file.close()
print("Hotovo -> Ulozeno")
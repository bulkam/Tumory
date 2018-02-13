# -*- coding: utf-8 -*-
"""
Created on Sun Feb 04 20:23:54 2018

@author: Mirab
"""


print("[INFO] START")

import keras
import keras.backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D 
from keras.layers import BatchNormalization, Concatenate
from keras.optimizers import SGD, RMSprop, Adam

import argparse
import h5py
import numpy as np

#import keras_data_reader as dr
import file_manager_metacentrum as fm
import CNN_experiment
import keras_callbacks

print("[INFO] Vse uspesne importovano - OK")


""" Nacteni argumentu """

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", help="Kolik se bude trenovat epoch", type=int, default=6)
parser.add_argument("--optimizer", help="Typ optimalizatoru", type=str, default="Nester")
parser.add_argument("--loss", help="Kriterialni funkce", type=str, default="categorical_crossentropy")
parser.add_argument("--LR", help="Pocatecni mira uceni", type=float, default=0.01)
parser.add_argument("--LR_change", help="Zpusob zmeny LR", type=str, default="det")

args = parser.parse_args()
print(args)
# nasteni argumentu
optimizer_label = args.optimizer
LR = args.LR
epochs = args.epochs
loss = args.loss
LR_change = args.LR_change

special_label = optimizer_label + "_SegNet2"
special_label = special_label + "_LR" + str(abs(int(np.log10(LR)))) + LR_change
special_label = special_label + "_" + str(epochs) + "epochs"


""" Nacteni dat """

#experiment_name = "no_aug_structured_data-liver_only"
experiment_name = "aug_structured_data-liver_only"
#experiment_name = "aug-ge+int_structured_data-liver_only"

experiment_foldername = "experiments/"+experiment_name
fm.make_folder(experiment_foldername)

hdf_filename = "datasets/processed/"+experiment_name+".hdf5"
hdf_file = h5py.File(hdf_filename, 'r')
train_data = hdf_file['train_data'][:]
train_labels = hdf_file["train_labels"][:]
val_data = hdf_file['val_data'][:]
val_labels = hdf_file["val_labels"][:]



""" Architektura """

inputs = Input(shape=(240, 232, 1))

# downsampling
xc1 = Conv2D(32, (3, 3), padding='same', activation='relu', strides=(1, 1))(inputs)
xc1b = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(xc1)
xc2 = Conv2D(64, (3, 3), padding='same', activation='relu', strides=(1, 1))(xc1b)
xc2b = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(xc2)
xmp3 = MaxPooling2D(pool_size=(2, 2))(xc2b)

xc4 = Conv2D(128, (3, 3), padding='same', activation='relu', strides=(1, 1))(xmp3)
xc4b = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(xc4)
xc5 = Conv2D(256, (3, 3), padding='same', activation='relu', strides=(1, 1))(xc4b)
xc5b = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(xc5)
xmp6 = MaxPooling2D(pool_size=(2, 2))(xc5b)

xc7 = Conv2D(256, (3, 3), padding='same', activation='relu', strides=(1, 1))(xmp6)
xc7b = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(xc7)
xc8 = Conv2D(512, (3, 3), padding='same', activation='relu', strides=(1, 1))(xc7b)
xc8b = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(xc8)
xmp9 = MaxPooling2D(pool_size=(2, 2))(xc8b)

# upsampling

xup1 = UpSampling2D(size=(2, 2), data_format=None)(xmp9)
concat1 = Concatenate(axis=-1)([xup1, xc8b])
xct2 = Conv2DTranspose(512, (3, 3), strides=(1, 1), padding='same', 
                       data_format=None, activation='relu')(concat1)
xct2b = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(xct2)
xct3 = Conv2DTranspose(256, (3, 3), strides=(1, 1), padding='same', 
                       data_format=None, activation='relu')(xct2b)
xct3b = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(xct3)

xup4 = UpSampling2D(size=(2, 2), data_format=None)(xct3b)
concat2 = Concatenate(axis=-1)([xup4, xc5b])
xct5 = Conv2DTranspose(256, (3, 3), strides=(1, 1), padding='same', 
                       data_format=None, activation='relu')(concat2)
xct5b = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(xct5)
xct6 = Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same', 
                       data_format=None, activation='relu')(xct5b)
xct6b = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(xct6)

xup7 = UpSampling2D(size=(2, 2), data_format=None)(xct6b)
xct8 = Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same', 
                       data_format=None, activation='relu')(xup7)
xct8b = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(xct8)
xct9 = Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same', 
                       data_format=None, activation='relu')(xct8b)
xct9b = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(xct9)

predictions = Conv2D(3, (1, 1), padding='same', activation='softmax')(xct9b)


""" Model """

metrics = ['accuracy']
# vytvoreni modelu
model = Model(inputs=inputs, outputs=predictions)

# vyber optimalizatoru
if optimizer_label.lower()=="sgd" and not "nester" in optimizer_label.lower():
    optimizer = SGD(lr=LR)#, clipvalue=0.5)
elif "rms" in optimizer_label.lower():
    optimizer = RMSprop(lr=LR, rho=0.9, decay=0.0)
elif "adam" in optimizer_label.lower():
    optimizer = Adam(lr=LR, beta_1=0.9, beta_2=0.999, decay=0.0)
else:
    optimizer = SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)

# kompilace modelu
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics)
print(model.summary())


""" Sprava souboru """
""" Pozor - jen pokud mam nejake specialni oznaceni """

if len(special_label) >= 1:
    if not experiment_foldername.endswith(special_label):
        experiment_foldername = experiment_foldername + "/" + special_label
else:
    special_label = "classic"
    if not experiment_foldername.endswith(special_label):
        experiment_foldername = experiment_foldername + "/" + special_label
    
fm.make_folder(experiment_foldername+"/logs")




""" FIT """

epochs = 6
class_weight = [0.1, 35.0, 4.0]
model.fit(train_data, train_labels, validation_data=(val_data, val_labels), 
          epochs=epochs, batch_size=8, shuffle=True,
          class_weight=class_weight, 
          callbacks=keras_callbacks.get_standard_callbacks_list(experiment_foldername+"/logs"))
          
model_filename = experiment_foldername+"/model.hdf5"
model.save(model_filename)

config = {"epochs": epochs,
         "class_weight": class_weight,
         "experiment_name": experiment_name,
         "experiment_foldername": experiment_foldername,
         "LR_begin": LR,
         "LR": float(K.eval(optimizer.lr)),
         "optimizer": str(optimizer),
         "loss": str(model.loss),
         "metrics": model.metrics}
fm.save_json(config, experiment_foldername+"/notebook_config.json")


""" Ohodnoceni """
CNN_experiment.evaluate_all(hdf_file, model, experiment_foldername)



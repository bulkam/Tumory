# -*- coding: utf-8 -*-
"""
Created on Sun Feb 04 12:05:06 2018

@author: Mirab
"""

import keras.callbacks
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger
from keras.callbacks import ReduceLROnPlateau, LambdaCallback, ProgbarLogger

import json


def LR_schedule(epoch, thrs=[1, 3, 5], basic_LR=0.01):
    
    LR = basic_LR
    
    for thr in thrs:
        if epoch >= thr:
            LR *= 0.1
        else:
            break
    print("LR -> ",LR)
    return LR


def LR_scheduler(factor=0.1, patience=2):
    LR_sh = LearningRateScheduler(LR_schedule)
#    LR_sh = ReduceLROnPlateau(monitor='val_loss', 
#                             factor=factor,
#                             patience=patience)
    return LR_sh


def checkpointer(path):
    if not path.endswith(".hdf5"):
        path = path + "model_checkpoint.hdf5"
    return ModelCheckpoint(filepath=path, verbose=1, save_best_only=True)


def progbar_logger():
    return ProgbarLogger(count_mode='samples')


class LearningRatePrinter(keras.callbacks.Callback):
    
    def __init__(self, verbose=0):
        super(LearningRatePrinter, self).__init__()
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr)
        print('\nLR: {:.6f}\n'.format(lr))


def LR_printer():
    return LearningRatePrinter()


def json_logging_callback(path):
    
    if not path.endswith(".json"):
        path = path + "/batches.log"
        
    json_log = open(path, mode='wt', buffering=1)
    
    lambda_call = LambdaCallback(on_batch_end=lambda batch, logs: json_log.write(json.dumps({'batch': batch,
                                                                                             'loss': float(logs['loss']), 
                                                                                             'acc': float(logs["acc"])}) + '\n'), 
                                 on_epoch_end=lambda epoch, logs: json_log.write(json.dumps({'epoch': epoch,
                                                                                             'loss': float(logs['loss']), 
                                                                                             'acc': float(logs["acc"])}) + '\n'), 
                                 on_train_end=lambda logs: json_log.close())
    return lambda_call


def csv_logger_fit(path):
    
    if not path.endswith("/csv_logger_fit"):
        path = path + "/csv_logger_fit"
        
    logger = CSVLogger(path, separator=',', append=False)
    
    return logger


def get_standard_callbacks_list(path):
    """ Vrati seznam callbacku, ktere budu volat pri metode fit modelu """
    
    callbacks_list = [json_logging_callback(path),
                     LR_scheduler(),
                     checkpointer(path),
                     csv_logger_fit(path)]
                     
    return callbacks_list

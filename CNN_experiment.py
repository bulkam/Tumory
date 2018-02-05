# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 16:06:33 2018

@author: mira
"""

print("[INFO] START")

import file_manager_metacentrum as fm
import CNN_evaluator
import CNN_boxes_evaluator
import numpy as np
import h5py

print("[INFO] Vse uspesne importovano - OK")


def save_results(model, test_data, test_labels, path="experiments/"):
    
    test_predicted_labels = model.predict(test_data)
    
    test_results_path = path if path.endswith(".hdf5") else path + "test_results.hdf5"
    hdf5_file = h5py.File(test_results_path , mode='w')
    hdf5_file.create_dataset("test_data", test_data.shape, np.int8)
    hdf5_file["test_data"][...] = test_data
    hdf5_file.create_dataset("test_labels", test_labels.shape, np.int8)
    hdf5_file["test_labels"][...] = test_labels
    hdf5_file.create_dataset("test_predictions", test_predicted_labels.shape, np.float)
    hdf5_file["test_predictions"][...] = test_predicted_labels
    hdf5_file.close()


def save_results_predicted(test_predicted_labels, hdf_file,
                           path="experiments/"):
    
    # nacteni dat
    test_data = hdf_file['test_data']
    test_labels = hdf_file["test_labels"]    
    
    test_results_path = path if path.endswith(".hdf5") else path + "test_results.hdf5"
    hdf5_file = h5py.File(test_results_path , mode='w')
    hdf5_file.create_dataset("test_data", test_data.shape, np.int8)
    hdf5_file["test_data"][...] = test_data
    hdf5_file.create_dataset("test_labels", test_labels.shape, np.int8)
    hdf5_file["test_labels"][...] = test_labels
    hdf5_file.create_dataset("test_predictions", test_predicted_labels.shape, np.float)
    hdf5_file["test_predictions"][...] = test_predicted_labels
    hdf5_file.close()
    


def save_results_predicted_reduced(test_predicted_labels, path="experiments/"):
    """ Ulozi jen test_predictions """
    test_results_path = path if path.endswith(".hdf5") else path + "test_results.hdf5"
    hdf5_file = h5py.File(test_results_path , mode='w')
    hdf5_file.create_dataset("test_predictions", test_predicted_labels.shape, np.float)
    hdf5_file["test_predictions"][...] = test_predicted_labels
    hdf5_file.close()



def evaluate_all(hdf_file, model, experiment_foldername, 
                 save_predictions=True):
    """ Ohodnoceni natrenovaneho modelu podle vsech moznych kriterii """

    # nacteni dat
    test_data = hdf_file['test_data']
    test_labels = hdf_file["test_labels"]
    
    # ohodnoceni Keras
    evaluation = model.evaluate(x=test_data, y=test_labels, batch_size=8)
    print(model.metrics_names, evaluation)
    # ulozeni do souboru
    eval_vocab = {}
    for i in range(len(model.metrics_names)):
        eval_vocab[model.metrics_names[i]] = evaluation[i]
    fm.save_json(eval_vocab, experiment_foldername+"/model_evaluation.json")
    
    # ohodnoceni vlastnimi metrikami
    print("[INFO] Probiha predikce testovacich dat...")
    test_predicted_labels = model.predict(test_data, batch_size=8).astype(np.float32)
    print("[INFO] Hotovo.")
    del model
    
    # ulozeni vysledku
    if save_predictions:
        save_results_predicted(test_predicted_labels, hdf_file,
                               path=experiment_foldername+"/test_results.hdf5")
    
    # --- Vlastni hodnotici metody ---
    my_eval_vocab = {}                            
    # Jaccard similarity
    JS = CNN_evaluator.evaluate_JS(test_labels, test_predicted_labels)
    my_eval_vocab.update(JS)
    # HoGovsky evaluate
    _, _, _, _, boxes_eval = CNN_boxes_evaluator.evaluate_nms_results_overlap(test_data, 
                                                                              test_labels, 
                                                                              test_predicted_labels)
    my_eval_vocab.update({"boxes": boxes_eval})
    # ulozeni mezivysledku
    fm.save_json(my_eval_vocab, experiment_foldername+"/evaluation.json")
    # accuracy per pixel
    ApP = CNN_evaluator.accuracy_per_pixel(test_labels, test_predicted_labels)
    my_eval_vocab["per_pixel_accuracy"] = ApP
    AMat_soft = CNN_evaluator.accuracy_matrix(test_labels,
                                              test_predicted_labels).tolist()
    my_eval_vocab["accuracy_matrix_soft"] = AMat_soft
    AMat_onehot = CNN_evaluator.accuracy_matrix(test_labels, 
                                                test_predicted_labels, 
                                                mode="onehot",
                                                batch_size=50).tolist()
    my_eval_vocab["accuracy_matrix_onehot"] = AMat_onehot
    # ulozeni vysledku
    fm.save_json(my_eval_vocab, experiment_foldername+"/evaluation.json")
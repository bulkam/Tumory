# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 16:06:33 2018

@author: mira
"""

print("[INFO] START")

import file_manager_metacentrum as fm
import CNN_evaluator
import CNN_boxes_evaluator

print("[INFO] Vse uspesne importovano - OK")



def evaluate_all(hdf_file, model, experiment_foldername):
    """ Ohodnoceni natrenovaneho modelu podle vsech moznyhc kriterii """

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
    test_predicted_labels = model.predict(test_data, batch_size=8)
    
    my_eval_vocab = {}
    # accuracy per pixel
    ApP = CNN_evaluator.accuracy_per_pixel(test_labels, test_predicted_labels)
    my_eval_vocab["per_pixel_accuracy"] = ApP
    AMat_soft = CNN_evaluator.accuracy_matrix(test_labels, 
                                              test_predicted_labels).tolist()
    my_eval_vocab["accuracy_matrix_soft": AMat_soft]
    AMat_onehot = CNN_evaluator.accuracy_matrix(test_labels, 
                                                test_predicted_labels, 
                                                mode="onehot").tolist()
    my_eval_vocab["accuracy_matrix_onehot": AMat_onehot]                                         
    # Jaccard similarity
    JS = CNN_evaluator.evaluate_JS(test_labels, test_predicted_labels)
    my_eval_vocab.update(JS)
    # HoGovsky evaluate
    _, _, _, _, boxes_eval = CNN_boxes_evaluator.evaluate_nms_results_overlap(test_data, 
                                                                              test_labels, 
                                                                              test_predicted_labels)
    my_eval_vocab.update({"boxes": boxes_eval})
    # ulozeni vysledku
    fm.save_json(my_eval_vocab, experiment_foldername+"/evaluation.json")
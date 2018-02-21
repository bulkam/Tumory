# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 13:09:14 2017

@author: Mirab
"""


import h5py
import numpy as np
import file_manager_metacentrum as fm
import CNN_evaluator

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys
import glob


def show_results(label, orig, result, lesion, 
                 label_label='Anotation', orig_label='Orig image',
                 result_label='Result label', lesion_label='Predicted lesion',
                 path="classification/Keras/results/comparation", index=0):
    """ Vykresli vysledky jako PNG obrazek a ulozi ho """
    
    f, [[ax11, ax12], [ax21, ax22]] = plt.subplots(2, 2)
    f.set_figheight(9)
    f.set_figwidth(9)

    ax11.imshow(label)
    ax11.set_title(label_label)
    ax11.grid()
    ax12.imshow(orig, cmap="gray")
    ax12.set_title(orig_label)
    ax12.grid()
    ax21.imshow(result)
    ax21.set_title(result_label)
    ax21.grid()
    ax22.imshow(lesion, cmap="gray")
    ax22.set_title(lesion_label)
    ax22.grid()
    #plt.show()
    
    plt.savefig(path+"/result_"+str("%.5d" % index)+".png")
    plt.close()


def generate_images(path, new_foldername="images", post_processing=False,
                    results_fname="test_results.hdf5"):
    """ Vygeneruje obrazky pro porovnani """
    
    images_path = path + "/" + new_foldername
    fm.make_folder(images_path)

    file = h5py.File(path + "/" + results_fname, 'r')
    
    test_data = file["test_data"]
    test_labels = file["test_labels"]
    test_predictions = file["test_predictions"]
    
#    dataname = fm.load_json(path+"/notebook_config.json")["experiment_name"]
#    file2 = h5py.File("datasets/processed/"+dataname+".hdf5", 'r')
#    test_data = file2["test_data"]       
    
    for index in range(test_data.shape[0]):
        if index % 100 == 0:
            print("Generuji se obrazky:", index, "/", test_data.shape[0])
        
        orig = test_data[index][:, :, 0].astype("uint8")
        label = test_labels[index].astype("uint8") * 255
        result = test_predictions[index].astype("float")
        lesion = np.argmax(result, axis=2)*127
                
        if post_processing:
            post, _ = CNN_evaluator.apply_morphology_operations(lesion, label)
            
            post[lesion!=0] += 1   
            blank = np.zeros(label.shape, dtype="uint8")
            blank[post[:]==0] = [255, 0, 0]
            blank[post[:]==1] = [0, 0, 255]
            blank[post[:]==2] = [0, 255, 0]
            
            show_results(label, orig, blank, result,
                         lesion_label="CNN output", 
                         result_label="Final result",
                         path=images_path, index=index) 
            
        else:
            show_results(label, orig, result, lesion,
                         path=images_path, index=index)
        #break
                         
    file.close()
    #file2.close()


def run_all(path_to_experiments="/experiments/aug_structured_data-liver_only/",
            results_fname="test_results.hdf5", post_processing=True):
    """ Pro vsechny experimenty vygeneruje PNG vysledky """
    
    folders = glob.glob(path_to_experiments+"*")
    
    for path in folders:
        print("[INFO] Generuji obrazky pro slozku: ", path)
        generate_images(path, post_processing=post_processing,
                        results_fname=results_fname)


if __name__ =='__main__': 
    
    path = "classification/Keras/results/test_results-aug_5epoch_structured_data-liver_only"
    path = "classification/Keras/results/test_results-no_aug_20epoch_structured_data-liver_only"
    path = "classification/Keras/results"
    path = "classification/Keras/experiments/aug_structured_data-liver_only/RMS_SegNet4_LRdet_5epochs_weighted-01-35-4"
    
    args = sys.argv
    if len(args) >= 2:
        path = str(sys.argv[1])

    if not "all" == path:
        generate_images(path, post_processing=True,
                        results_fname="test_results.hdf5")
    else:
        run_all()
    #test_results-5epoch_aug_structured_data
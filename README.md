# Tumory
Liver lesions detector

## HoG + SVM

### Dataset

#### Prepare your dataset

##### Create dataset manually
1. Insert positive frames into -> **dataset/processed/orig_images**
2. Insert negative frames into -> **dataset/processed/negatives**
3. Insert test frames into -> **dataset/processed/test_images**

Images should be in format .png or .pklz.

##### OR
1. Insert CT data (.pklz format) into -> **CTs/** 
2. Extract slices from CT data (.pklz format) 
> CTs/ python data_augmentation.py
3. Copy dataset into target folders. 
> python file_manager.py

#### Feature extraction
> python extract_feature_vectors.py

This command extracts feature vectors into **classification/training_data/hog_features.json** file.

### Model fitting
> python run_SVM.py *args*

##### Args:
- if "hnm" is in the list of *args*:
-> 1. SVM model will be fitted
-> 2. HNM method will be performed
-> 3. SVM model will be fitted again and saved into **classification/classifiers/**
- if "train" is in the list of *args*, the SVM model will be fitted and saved into **classification/classifiers/**
- if "test" is in the list of *args*, the trained SVM model will be used for the detection of liver lesions in the test images. 
- if "evaluate" is in the list of *args*, the SVM model will be evaluated on the test images

Note 1: _Use "train" argument only if you do not want to perform HNM or if you have hard negatives already extracted. Otherwise, use "hnm" argument for classifier training._

Note 2: _Use following commang:_
> python run_SVM.py  hnm  test  evaluate 

_for perform HNM, fit the SVM model, detect liver lesions in the test images and evaluate the trained SVM model on them_

Note 3: _if no argument is present, the script will run as the same as with only a single argument "train"_

### Prediction
> python run_SVM.py  test

##### This command:
1. Loads trained SVM model from **classification/classifiers/**.
2. Performs detection of liver lesions in the test images using this trained model.
3. Stores test results as **classification/results/test_results.json**
4. Stores test results after NMS as **classification/results/results_nms.json**
5. Generates resulting images with detected bounding boxes into **classification/results/PNG_results/**

### Evaluation
> python run_SVM.py  evaluate

##### This command:
1. Loads test results after NMS from the file **classification/results/results_nms.json**
2. Evaluates theese test results using several metrics computed from the overlap between detected bounding boxes and ground truth annotations (lesions).
3. Stores the evaluation results as **classification/evaluation/nms_overlap_evaluation.json** 

## CNN
### Dataset
1. Insert CT data (.pklz format) into -> **CTs/** 
2. Generate image data.
> CTs/ python kerasdata_maker.py

3. Create dataset (.hdf5 format) from generated images.
> python keras_dataset_maker.py

### Model fitting
Choose architecture script name as SegNet*ArchitectureType*.py
> python *script_name*.py

This command fits a model using the selected architecture and evaluates it with several metrics.

The model will be saved in the folder **experiments**/*dataset_name*/*experiment_name*/ as **model.hdf5**.

### Load fitted model and evaluate it
> python ReEvaluate_trained_model.py   *path_to_model_file*

- This command loads existing model and evaluate it on the training set. Dataset name is simply obtained from the defined path to the model.

- Argument *path_to_model_file* should contain the name of the folder where the model is stored and it has to be defined.

- The resulting predictions will be saved in this folder with the name **test_results.hdf5**.

### Visualize results predicted by CNN
> python keras_result_explorer.py    *path_to_model_file*

##### This command:
1. loads ***path_to_model_file*/test_results.hdf5**
2. extracts images
3. visualizes results and stores them into ***path_to_model_file*/images/**

Argument *path_to_model_file* should contain the name of the folder where the model is stored and it has to be defined.


## Configuration

Many settings such as paths to saving/loading some files or parameters of the most of used methods are writen in two configuration files:
1. **CTs/Configuration/config.json** - contains paths and parameters which are associated with the process of creating dataset.
2. **configuration/CT.json** - contains paths and parameters mostly associated with the methods used for the image processing, feature extraction and classification.

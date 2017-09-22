# -*- coding: utf-8 -*-
"""
Created on Sun May 21 18:49:40 2017

@author: mira
"""

import classifier as clas
import feature_extractor as fe
import data_reader as dr

import os
import cv2

import skimage
from skimage.feature import hog as hogg
from skimage import exposure
from skimage.filters import roberts, sobel, scharr, prewitt
from matplotlib import pyplot as plt
import scipy
import numpy as np

from sklearn.feature_extraction.image import extract_patches_2d


def test_hogs():
    hog = fe.HOG()
    xpath = hog.config_path
    config = hog.dataset.config
    win = hog.sliding_window_size
    
    data = hog.dataset
    print hog.sliding_window_size
    print len(data.orig_images)
    #print hog.dataset.load_annotated_images()
    TM = hog.extract_features()
    
    return TM


def show_plot_in_new_figure(data):
    """ Vykresli graf v novem okne """
    
    plt.figure(figsize = (30,10))
    plt.ylim(-0.3, 0.3)
    plt.plot(list(data), 'b', lw=1)
    plt.grid()
    plt.show()


def show_image_in_new_figure(img):
    """ Vykresli obrazek v novem okne """
    
    plt.figure()
    skimage.io.imshow(img, cmap = 'gray')
    plt.show()


def draw_keypoints(img, keypoints):
    """ Vykresli dane keypointy do obrazku """
    
    result = cv2.drawKeypoints(img, keypoints[:], None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    
    show_image_in_new_figure(result)


def draw_hogs(img):
    """ Vykresli HoGy do obrazku """

    fd, hog_img = hogg(img, orientations=4, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualise=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10), sharex=True, sharey=True)
    
    ax1.axis('off')
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('Vstupni obrazek')
    ax1.set_adjustable('box-forced')
    
    hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 0.02))
    
    ax2.axis('off')
    ax2.imshow(hog_img_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram Orientovanych Gradientu')
    ax1.set_adjustable('box-forced')
    plt.show()
    
    
# TODO: zkouset 
def preprocess_image(img):
    """ Provede zakladni predzpracovani obrazku """
    
    roi = cv2.resize(img, tuple(hog.sliding_window_size), interpolation=cv2.INTER_AREA)
    #show_image_in_new_figure(roi)
    #roi = scipy.ndimage.gaussian_filter(roi, sigma=1)
    roi = cv2.bilateralFilter(roi.astype("uint8"), 5, 35, 35)
    #roi = sobel(roi)
    return roi
    

def show_hogs(imgname, hog):
    
    img = dataset.load_image(imgname)

    roi = preprocess_image(img)
    
    # extrakce vektoru priznaku
    feature_vect = hog.skimHOG(roi)
    feature_vect = hog.extract_single_feature_vect(roi)[0]
    
    #show_image_in_new_figure(roi)
    #show_plot_in_new_figure(feature_vect)
    #draw_hogs(roi)
    
    return feature_vect


def show_keypoints(imgname, to_save=False, name="image"):
    """ Jen zkouseni nastaveni keypoint detectoru """
    
    img = dataset.load_image(imgname).astype("uint8")
    
    roi = preprocess_image(img)
    
    descriptor_list = list()
    
    #feature_detector = cv2.SIFT(nfeatures=4, nOctaveLayers=2, contrastThreshold=0.01, edgeThreshold=0.04, sigma=1.6)
    feature_detector = cv2.FeatureDetector_create("ORB")
    extractor = cv2.DescriptorExtractor_create("ORB")
    
    keypoints = feature_detector.detect(roi)
    #keypoints, descriptor = extractor.compute(roi, keypoints)
#    print type(descriptor), descriptor.shape
#    descriptor = np.zeros((7, 128)) if descriptor is None else descriptor
#    
#    descriptor_list.append([imgname, descriptor.astype('float32')]) #descriptory maji stejne delky, ale je jich ruzny pocet matice N x 158 napr.
#    print len(descriptor_list), len(descriptor_list[0]), len(descriptor_list[0][0])
    #show_plot_in_new_figure(descriptor)
    draw_keypoints(roi, keypoints)
    if to_save: plt.savefig(name)


def show_SIFTs(imgname, sift):
    img = dataset.load_image(imgname)
    
    roi = preprocess_image(img)
    
    # extrakce vektoru priznaku
    print np.max(img.astype("uint8"))
    feature_vect = sift.extract_single_feature_vect(img.astype("uint8"))
   
    #print feature_vect
    
    show_image_in_new_figure(roi, to_save=True)
    show_plot_in_new_figure(feature_vect)


def visualize_data(pos, neg, n=-1, draw_all=False):
    
    P = np.vstack(pos)[:, 0:n]
    N = np.vstack(neg)[:, 0:n]
    
    mP = np.mean(P, axis = 0)
    mN = np.mean(N, axis = 0)
    
    varP = np.var(P, axis = 0)
    varN = np.var(N, axis = 0)
    
    hP = mP + varP
    lP = mP - varP
    hN = mN + varN
    lN = mN - varN
    
    plt.figure()
    plt.ylim(-1, 1)
    plt.plot(mP, color='y')
    plt.plot(mN, color='b')
    plt.fill_between(np.arange(len(mP)), lP, hP, where=hP >= lP, facecolor='red', interpolate=True)
    plt.fill_between(np.arange(len(mP)), lN, hN, where=hN >= lN, facecolor='green', interpolate=True)
    plt.grid()
    plt.show()
    
    if draw_all:
        plt.figure()
        
        for n in N:
            plt.plot(n, 'g')
        
        for p in P:
            plt.plot(p, 'r')
            
        plt.show()
    

if __name__ =='__main__':  
    
    # SIFT
    sift = fe.SIFT()
    # HoG
    hog = fe.HOG(orientations=12, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    
    config = hog.dataset.config
    dataset = hog.dataset
    #print hog.count_sliding_window_size(boxes_path=dataset.annotations_path)
    
    positives = [config["frames_positives_path"]+imgname for imgname in os.listdir(os.path.dirname(os.path.abspath(__file__))+"/"+config["frames_positives_path"]) if imgname.endswith('.pklz')]
    negatives = [config["frames_negatives_path"]+imgname for imgname in os.listdir(os.path.dirname(os.path.abspath(__file__))+"/"+config["frames_negatives_path"]) if imgname.endswith('.pklz')]
    
    indexes = [18, 129, 146, 222, 21, 64, 111]
    indexes = range(min(len(positives), len(negatives))//10)
    
    positive_feature_vects = list()
    negative_feature_vects = list()
    
    for p in indexes:
        positive = positives[p]
        positive_feature_vects.append(show_hogs(positive, hog))
        #show_SIFTs(positive, sift)
        #show_keypoints(positive, to_save=True, name="extractor_test_results/positives/positive"+str(p))
    
    
    for n in indexes:
        negative = negatives[n]
        negative_feature_vects.append(show_hogs(negative, hog))
        #show_SIFTs(positive, sift)
        #show_keypoints(negative, to_save=True, name="extractor_test_results/negatives/negative"+str(n))
        
    visualize_data(positive_feature_vects, negative_feature_vects,
                   draw_all=True, n=12)
    
    """
    img = dataset.load_image(positives[111])
    patches = extract_patches_2d(img, (24, 28), max_patches = 4)
    for patch in patches:
        show_image_in_new_figure(patch)
    """





    
    
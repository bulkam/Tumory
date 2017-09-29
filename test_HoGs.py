# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 18:01:47 2017

@author: mira
"""

import classifier as clas
import feature_extractor as fe
import data_reader as dr
import file_manager as fm

import re
import os
import cv2
import copy

import skimage
from skimage.feature import hog as hogg
from skimage import exposure, data
#from skimage.filters import roberts, sobel, scharr, prewitt
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy
import numpy as np

from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import PCA as PCA
from sklearn.decomposition import TruncatedSVD as DEC


def show_plot_in_new_figure(data, ylim=(-0.3, 0.3),
                            to_save=False, fname="extractor_test_results/result.png"):
    """ Vykresli graf v novem okne """
    
    plt.figure(figsize = (30,10))
    plt.ylim(ylim)
    plt.plot(list(data), 'b', lw=1)
    plt.grid()
    plt.show()
    
    if to_save:
        plt.savefig(fname)


def show_image_in_new_figure(img, to_save=False, fname="extractor_test_results/result.png"):
    """ Vykresli obrazek v novem okne """
    
    plt.figure()
    skimage.io.imshow(img, cmap = 'gray')
    plt.show()
    
    if to_save:
        plt.savefig(fname)


def reduce_single_vector_dimension(vect):
    """ Nacte model PCA a aplikuje jej na jediny vektor """

    # aplikace ulozeneho PCA
    reduced = pca.transform(vect)      # redukuje dimenzi vektoru priznaku

    return reduced


def extract_single_feature_vect(gray):
    """ Vrati vektor priznaku pro jedek obrazek """

    hist, hog_img = skimHOG(gray)
    reduced = reduce_single_vector_dimension(hist)

    return reduced, hog_img

    
def show_hogs(imgname, hog, to_draw=False):
    
    img = dataset.load_image(imgname)

    roi = preprocess_image(img)
    
    # extrakce vektoru priznaku
    feature_vect, hog_img = skimHOG(roi)
    #feature_vect, hog_img = extract_single_feature_vect(roi)[0]
    
    #show_image_in_new_figure(roi)
    #show_plot_in_new_figure(feature_vect)
    if to_draw: draw_hogs(roi, hog_img, feature_vect)
    
    return feature_vect, hog_img


def reduce_dimension(positives, negatives, to_return=True, fv_len=10,
                     new_pca=True):
    """ Aplikuje PCA a redukuje tim pocet priznaku """

    features = dict()       
    
    # namapovani na numpy matice pro PCA
    X = np.vstack((np.vstack(positives), np.vstack(negatives)))
    Y = np.vstack((np.vstack([1]*len(positives)), np.vstack([-1]*len(negatives)))) 
    
    print "Data shape: ", X.shape, Y.shape, len(positives[0])
    
    # PCA
    if new_pca or pca is None:
        pca = PCA(n_components=fv_len)   # vytvori PCA
        #pca = DEC(n_components=fv_len)   # vytvori PCA
        pca.fit(X, Y)
    
    reduced = pca.transform(X)      # redukuje dimenzi vektoru priznaku
    
    # znovu namapuje na zavedenou strukturu
    features = list(reduced)
    
    # ulozeni PCA
    #dataset.save_obj(pca, self.PCA_path+"/PCA_"+self.descriptor_type+".pkl")

    if to_return: return pca, features

    
def draw_hogs(img, hog_img, vect, rescale=True, fname="hog_plot.png"):
    """ Vykresli HoGy do obrazku """
    """
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16, 10), sharex=True, sharey=True)
    
    ax1.axis('off')
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('Vstupni obrazek')
    ax1.set_adjustable('box-forced')
    
    hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 0.02))
    
    ax2.axis('off')
    ax2.imshow(hog_img_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram Orientovanych Gradientu')
    ax2.set_adjustable('box-forced')
    
    plt.show()"""
    
    fig = plt.figure(figsize=(16, 10))

    gs = GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, :1])
    ax2 = plt.subplot(gs[0, -1])
    ax3 = plt.subplot(gs[1, :])
    
    ax1.axis('off')
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('Vstupni obrazek')
    #ax1.set_adjustable('box-forced')
    
    hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 0.02)) if rescale else hog_img
    
    ax2.axis('off')
    ax2.imshow(hog_img_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram Orientovanych Gradientu')
    #ax2.set_adjustable('box-forced')
    
    ax3.plot(vect)
    ax3.grid()
        
    plt.show()
    plt.savefig(foldername+"/hog_plots/"+fname)
    plt.savefig(parentname+"/hog_plots/"+fname+"/"+childname+".png")
    

def visualize_data(pos, neg, n_features=12,
                   var_scale=1, draw_all=False, each_data=1):
    
    P = np.vstack(pos)[:, 0:n_features]
    N = np.vstack(neg)[:, 0:n_features]
    
    mP = np.mean(P, axis = 0)
    mN = np.mean(N, axis = 0)
    
    varP = np.var(P, axis = 0)
    varN = np.var(N, axis = 0)
    
    hP = mP + varP * var_scale
    lP = mP - varP * var_scale
    hN = mN + varN * var_scale
    lN = mN - varN * var_scale
    
    plt.figure()
    plt.ylim(max(max(hP), max(hN)), min(min(lP), min(lN)))
    plt.plot(mP, color='y')
    plt.plot(mN, color='b')
    plt.fill_between(np.arange(len(mP)), lP, hP, where=hP >= lP, facecolor='red', interpolate=True)
    plt.fill_between(np.arange(len(mP)), lN, hN, where=hN >= lN, facecolor='green', interpolate=True)
    plt.grid()
    plt.show()
    plt.savefig(foldername+"/data/features_filled_"+str(fv_length)+".png")
    plt.savefig(parentname+"/data/features_filled/"+childname+"_fvlen="+str(fv_length)+".png")
    
    if draw_all:
        plt.figure()
        plt.ylim(-1, 1)
        
        for i, p in enumerate(P):
            if i % each_data == 0:
                plt.plot(p, 'r')
                plt.plot(N[i], 'g')  
        
        plt.grid()
        plt.show()
        plt.savefig(foldername+"/data/features_"+str(fv_length)+".png")
        plt.savefig(parentname+"/data/features_all/"+childname+"_fvlen="+str(fv_length)+".png")


def visualize_feature_pairs(pos, neg, features = (0, 1), n_features=-1,
                            var_scale=1, draw_all=False, each_data=1):
    P = np.vstack(pos)[:, :]
    N = np.vstack(neg)[:, :]
    
    print P[:, features[0]].shape
    
    #P = np.hstack((P[:, features[0]], P[:, features[1]]))
    #N = np.hstack((N[:, features[0]], N[:, features[1]]))
    
    print P.shape
    xmin = min(np.min(P[:, features[0]]), np.min(N[:, features[0]]))
    xmax = max(np.max(P[:, features[0]]), np.max(N[:, features[0]]))
    ymin = min(np.min(P[:, features[1]]), np.min(N[:, features[1]]))
    ymax = max(np.max(P[:, features[1]]), np.max(N[:, features[1]]))
                                                                
    plt.figure()
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))
    plt.scatter(P[:, features[0]], P[:, features[1]], color='r')
    plt.scatter(N[:, features[0]], N[:, features[1]], color='g')
    plt.grid()
    plt.savefig(parentname+"/data/"+childname+"both.png")
    plt.savefig(parentname+"/data/pair/both/"+childname+".png")
    plt.savefig(foldername+"/data/pair_both.png")
    plt.show()
    
    
    plt.figure()
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))
    plt.scatter(P[:, features[0]], P[:, features[1]], color='r')
    plt.grid()
    plt.savefig(parentname+"/data/pair/"+childname+"POS.png")
    plt.savefig(foldername+"/data/pair_pos.png")
    plt.show()
    
    plt.figure()
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))
    plt.scatter(N[:, features[0]], N[:, features[1]], color='g')
    plt.grid()
    plt.savefig(parentname+"/data/pair/"+childname+"NEG.png")
    plt.savefig(foldername+"/data/pair_neg.png")
    plt.show()
    

# TODO: zkouset 
def preprocess_image(img):
    """ Provede zakladni predzpracovani obrazku """
    
    roi = cv2.resize(img, tuple(config["sliding_window_size"]), interpolation=cv2.INTER_AREA)
    
    roi = cv2.bilateralFilter(roi.astype("uint8"), 9, 35, 35)
    
    # histogram
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    #roi = clahe.apply(roi.astype("uint8"))
    #roi = cv2.equalizeHist(roi.astype("uint8"))
    
    return roi


def skimHOG(roi):
    
    img = roi
    
    hist, hog_img = hogg(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block, visualise=True)
    
    hist[hist<0] = 0
    
    return hist, hog_img  
    
def show_hogs2(img, to_draw=False, fname="hog_plot"):

    roi = preprocess_image(img)
    
    # extrakce vektoru priznaku
    feature_vect, hog_img = skimHOG(roi)
    #feature_vect, hog_img = extract_single_feature_vect(roi)[0]
    
    #show_image_in_new_figure(roi)
    #show_plot_in_new_figure(feature_vect)
    if to_draw: draw_hogs(roi, hog_img, feature_vect, rescale=False, fname=fname)
    
    return feature_vect, hog_img


def get_orig_image(imgname, config):
    
    maskname = fm.get_maskname(imgname, config)
    orig_imgname = re.sub('masks', 'orig_images', maskname)
    return dr.load_image(orig_imgname)

    
def color_background(imgname, mode='pos', to_draw=False, to_color=False):
    
    frame = None
    mask_frame = None
    x, h, y, w = [0]*4
    
    if mode in ['p', 'P', 'pos', 'POS', 'Pos']:
        x, h, y, w = fm.get_bb_from_imgname(imgname)
        img = get_orig_image(imgname, config)
        frame = img[x:h, y:w]
        
        mask = fm.get_mask(imgname, config)
        mask_frame = mask[x:h, y:w]
        
    elif mode in ['H', 'h', 'HNM', 'hnm']:
        x, h, y, w = fm.get_bb_from_imgname(imgname)
        frame = dr.load_image(imgname)
        
        mask = fm.get_mask(imgname, config)
        mask_frame = mask[x:h, y:w]
    
    elif mode in ["NEG", "neg", "n", "N", "Neg"]:
        frame = dr.load_image(imgname)
        mask_frame = np.ones(frame.shape)*2
    
    #show_image_in_new_figure(img)
    #show_image_in_new_figure(mask)

#    show_image_in_new_figure(frame)
#    show_image_in_new_figure(mask_frame)
   
    blur = copy.copy(frame)
    
    """ Zde se prebarvuje okoli """
    # TODO: testovat
    if to_color:
        # obarveni pozadi
        liver = np.mean(frame[mask_frame>0])
        # pripadne vykresleni
        if to_draw: 
            show_hogs2(frame, to_draw=to_draw, fname=fm.get_imagename(imgname))
        # prebarveni okoli
        blur[mask_frame==0] = liver
        blur = cv2.GaussianBlur(blur,(3,3), 0)
        blur[mask_frame>0] = frame[mask_frame>0]
    
    """ ---- Konec prebarvovani ----- """
    
    # hog deskriptor
    hist, hog_img = show_hogs2(blur, to_draw=to_draw, fname=fm.get_imagename(imgname)+col)
    
    #show_image_in_new_figure(blur)
        
    return hist


if __name__ =='__main__':
    
    # Nastaveni modu
    
    explore_data = bool(1)        
    show_hog_images = bool(0)
    
    dataset = dr.DATAset()
    dataset.create_dataset_CT()
    config = dataset.config
    
    pca = None
    
    pos_test_path = "datasets/frames/testable/positives/"
    neg_test_path = "datasets/frames/testable/negatives/"
    hnm_test_path = "datasets/frames/testable/HNM/"
    
    pos_path = "datasets/PNG/datasets/frames/positives/"
    neg_path = "datasets/PNG/datasets/frames/negatives/"
    hnm_path = "datasets/PNG/datasets/frames/HNM/"
    
    positives = [pos_path + imgname for imgname in os.listdir(pos_path) if imgname.endswith('.png') and not ('AFFINE' in imgname)]
    hnms = [hnm_path + imgname for imgname in os.listdir(hnm_path) if imgname.endswith('.png') and not ('AFFINE' in imgname)]
    negatives = [neg_path + imgname for imgname in os.listdir(neg_path) if imgname.endswith('.png') and not ('AFFINE' in imgname)]

    positives = [pos_path + imgname for imgname in os.listdir(pos_path) if imgname.endswith('.png')]
    hnms = [hnm_path + imgname for imgname in os.listdir(hnm_path) if imgname.endswith('.png')]
    negatives = [neg_path + imgname for imgname in os.listdir(neg_path) if imgname.endswith('.png')]
        
    n_images = -1#min(len(positives), len(negatives))
    n_each = 10
    
    positives = positives[:n_images]
    hnms = hnms[:n_images]
    negatives = negatives[:n_images]
    hnms0 = list()
    #negatives = hnms
    
    """ Nastaveni parametru """
    
    orientations=16
    pixels_per_cell=(8, 8)
    cells_per_block=(2, 2)
    fv_length = 10
    
    coloring=bool(1)
    col = "_colored" if coloring else ""
    
    print "Nastaveny parametry."
    
    manager = fm.Manager()
    parentname = "extractor_test_results/HoG"
    childname = "ori="+str(orientations)+"_ppc="+str(pixels_per_cell[0])+"_cpb="+str(cells_per_block[0])+col
    foldername = parentname+"/"+childname
    manager.make_folder(foldername+"/data")
    manager.make_folder(foldername+"/hog_images")
    manager.make_folder(foldername+"/hog_plots")
    
    # Spocteni HoG features
    if explore_data:
    
        P = list()
        N = list()
        
        for p, positive in enumerate(positives[:]):
            if p % 500 == 0:
                print p
            if p % n_each == 0:
                try:
                    P.append(color_background(positive, mode="POS", to_color=coloring))
                except:
                    pass
        
        for n, negative in enumerate(negatives[:]):
            if n % 500 == 0:
                print n
            if n % n_each == 0:
                try:
                    N.append(color_background(negative, mode='n', to_color=coloring))
                except:
                    pass
        
        for n, negative in enumerate(hnms[:]):
            if n % 500 == 0:
                print n
            if n % n_each == 0:
                try:
                    N.append(color_background(negative, mode='hnm', to_color=coloring))
                except:
                    pass
        
        print len(positives), len(P)
        print len(negatives), len(hnms), len(N)

        # vizualizace dat o dimenzi 2
        
        pca, feature_vects = reduce_dimension(P, N,
                                              fv_len=2, 
                                              new_pca=True)
    
        pos = feature_vects[:len(P)]
        neg = feature_vects[len(N):]
        
        visualize_feature_pairs(pos, neg,
                                features=(0, 1), n_features=-1,
                                var_scale=1, draw_all=False, each_data=1)
        
        # ted jen feature vektory o libovolne delce
        
        pca, feature_vects = reduce_dimension(P, N,
                                              fv_len=fv_length, 
                                              new_pca=True)
        pos = feature_vects[:len(P)]
        neg = feature_vects[len(N):]
        visualize_data(pos, neg,
                       draw_all=True, each_data=20, n_features=-1, var_scale=1)
                   
    """ -------------- Konec testovani dat ----------- """
    
    if show_hog_images:
        
        positives = [pos_test_path + imgname for imgname in os.listdir(pos_test_path) if imgname.endswith('.png')]
        hnms = [hnm_test_path + imgname for imgname in os.listdir(hnm_test_path) if imgname.endswith('.png')]
        negatives = [neg_test_path + imgname for imgname in os.listdir(neg_test_path) if imgname.endswith('.png')]
        
        P_to_draw = list()
        N_to_draw = list()
        
        n_each = 1
        
        for p, positive in enumerate(positives[:]):
            if p % 500 == 0:
                print p
            if p % n_each == 0:
                try:
                    P_to_draw.append(color_background(positive, to_draw=True, 
                                                      mode="POS", to_color=coloring))
                except:
                    pass
        
        for n, negative in enumerate(negatives[:]):
            if n % 500 == 0:
                print n
            if n % n_each == 0:
                try:
                    N_to_draw.append(color_background(negative, to_draw=True, 
                                                      mode='n', to_color=coloring))
                except:
                    pass
        
        for n, negative in enumerate(hnms[:]):
            if n % 500 == 0:
                print n
            if n % n_each == 0:
                try:
                    N_to_draw.append(color_background(negative, to_draw=True, 
                                                      mode='hnm', to_color=coloring))
                except:
                    pass
    
    
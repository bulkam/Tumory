# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 19:13:19 2017

@author: mira
"""

import numpy as np
import cv2

from matplotlib import pyplot as plt
from skimage.feature import hog

import skimage.io
import skimage.exposure as exposure

import os
import copy

import scipy
from scipy import io
from scipy.cluster.vq import *

from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import PCA

import random
import cPickle

import data_reader


def liver_coverage(mask_frame):
    """ Vrati procentualni zastoupení jater ve framu """
    
    # spocteni pixelu
    total_pixels = mask_frame.shape[0] * mask_frame.shape[1]
    liver_pixels = np.sum((mask_frame >= 1).astype(int))
    # spocteni pokryti obrazku jatry
    return float(liver_pixels) / total_pixels
    

def get_mask_frame(mask, bounding_box):
    """ Z daneho scalu a souradnic exrahuje okenko masky """
    
    (x, h, y, w) = bounding_box
    return mask[x:h, y:w]


def artefact_coverage(mask_frame):
    """ Vrati zastoupeni nalezu ve snimku """
    
    # spocteni pixelu
    total_pixels = mask_frame.shape[0] * mask_frame.shape[1]
    liver_pixels = np.sum((mask_frame == 1).astype(int))
    # spocteni pokryti obrazku jatry
    return float(liver_pixels) / total_pixels


class Extractor(object):
    
    def __init__(self, configpath="configuration/", configname="CT.json"):
        
        self.config_path = configpath + configname
        self.dataset = data_reader.DATAset(configpath, configname)
        
        self.dataset.create_dataset_CT()
        
        self.PCA_path = self.dataset.config["PCA_path"]
        self.PCA_object = None
        
        self.feature_vector_length = self.dataset.config["feature_vector_length"]
        
        self.sliding_window_size = self.dataset.config["sliding_window_size"]
        self.bounding_box_padding = self.dataset.config["bb_padding"]

        self.image_preprocessing = bool(self.dataset.config["image_preprocessing"])
        self.flip_augmentation = bool(self.dataset.config["flip_augmentation"])
        self.intensity_augmentation = bool(self.dataset.config["intensity_augmentation"])
        
        self.descriptor_type = str()
        
        self.features = dict()


    def get_roi(self, img, bb, padding=None, new_size=None, 
                image_processing=True):
        """ Podle bounding boxu vyrizne z obrazku okenko """
        
        if padding is None: padding = self.bounding_box_padding
        if new_size is None: new_size = self.sliding_window_size
        
        (i, h, j, w) = bb
        (i, j) = (max(i-padding, 0), max(j-padding, 0))
        roi = img[i:h+padding, j:w+padding]
        
        # zmeni velikost regionu a vrati ho
        roi = cv2.resize(roi, new_size[::-1], interpolation = cv2.INTER_AREA)
        # intenzitni transformace
        if image_processing: roi = self.image_processing(roi)                  
        
        return roi
    
    # TODO: zkouset
    def apply_image_processiong(self, roi):
        """ Aplikuje na obraz vybrane metody zpracovani obrazu """
        
        out = copy.copy(roi.astype("uint8"))
        # bilatelarni transformace
        out = cv2.bilateralFilter(out, 9, 35, 35)
        # vyuziti celeho histogramu
        out = exposure.rescale_intensity(out)

        # vrati vysledek
        return out
        
    
    def image_processing(self, rois):
        """ Predzpracovani obrazu """
        
        if len(rois.shape) >= 3:
            new_rois = list()
            
            for roi in rois:
                new_rois.append(self.apply_image_processiong(roi))
                
            return new_rois
        
        else:
            return self.apply_image_processiong(rois)

    
    def flipped_rois_generator(self, roi):
        """ Vrati ruzne otoceny vstupni obrazek 
        - nejdrive 4 flipy a pote to same pro transpozici """
        
        roi_tmp = copy.copy(roi)
        
        for i in xrange(2):
            yield roi_tmp
            roi_tmp = np.flip(roi_tmp, axis=0)
            yield roi_tmp
            roi_tmp = np.flip(roi_tmp, axis=1)
            yield roi_tmp
            roi_tmp = np.flip(roi_tmp, axis=0)
            yield roi_tmp
            roi_tmp = copy.copy(roi.T) 
    
    
    def intensity_transformed_rois_generator(self, roi, intensity_transform=False):
        """ Aplikuje na obraz ruzne intenzitni transformace """
        
        # originalni intenzita
        yield roi
        
        if intensity_transform:
                        
            # TODO: zasumeni dat - zvolit ty scaly -> v configu
            scales = self.dataset.config["intensity_augmentation_noise_scales"]                
            for scale in scales:
                
                # pricteni aditivniho sumu
                noised_roi = roi + np.random.normal(loc=0.0, scale=scale, size=roi.shape)
                # omezeni na interval
                noised_roi[noised_roi >= 255] = 255
                noised_roi[noised_roi <= 0] = 0
                
                yield noised_roi.astype("uint8")
            
            
    def multiple_rois_generator(self, rois):
        """ Vrati puvodni obrazek a pote nekolik jeho modifikaci """
        
        for roi in rois:
            
            for roi_intensity in self.intensity_transformed_rois_generator(roi, intensity_transform=self.intensity_augmentation):
                
                if not self.flip_augmentation: 
                    yield roi_intensity
                    continue
                    
                for roi_tmp in self.flipped_rois_generator(roi_intensity):
                    yield roi_tmp                        # puvodni obrazek

   
    def pyramid_generator(self, img, scale=1.5, min_size=(30, 30)):
        """ Postupne generuje ten samy obrazek s ruznymi rozlisenimy """
        
        # nejdrive vrati obrazek v puvodni velikosti
        yield img
        
        min_h, min_w = min_size
        
        # pote zacne vracet zmensene obrazky
        while True:
            img = scipy.misc.imresize(img, 1.0/scale)        # zmensi se obrazek
            height, width = img.shape[0:2]
            
            # pokud je obrazek uz moc maly, zastavi se proces
            if (height < min_h) or (width < min_h):    
                break
            
            # vrati zmenseny obrazek
            yield img


    def sliding_window_generator(self, img, step=4, window_size=[32,28], 
                                 image_processing=True):
        """ Po danych krocich o velikost step_size prostupuje obrazem 
            a vyrezava okenko o velikost window_size """
            
        (height, width) = img.shape[0:2]
        (win_height, win_width) = window_size
        h = 0
        while True:
            w = 0
            while True:
                box = [h, h+win_height, w, w+win_width]
                roi = self.image_processing(img[h:h+win_height, w:w+win_width]) if image_processing else img[h:h+win_height, w:w+win_width]
                yield (box, roi)
                
                w += step
                if w+step >= width:
                        break
            h += step
            if h+step >= height:
                    break

       
    def count_sliding_window_size(self, boxes_path):
        """ Spocita velikost sliding window na zaklade prumerne velikost b.b. """
        
        boxes = self.dataset.precti_json(boxes_path)
        widths = []
        heights = []
        
        for box in boxes.keys():
            for i in xrange(len(boxes[box])):
                # Prida se sirka a vyska do seznamu
                (y, h, x, w) = boxes[box][i]
                heights.append(h-y)
                widths.append(w-x)
        	
        # ze seznamu se spocita prumer jak pro vysku, tak pro sirku
        avg_height, avg_width = np.mean(heights), np.mean(widths)
        print " - prumerna sirka: ", avg_width
        print " - prumerna vyska: ", avg_height
        print " - pomer stran: ", avg_width / avg_height
        
        width = int(np.round(avg_width/8)*4)
        height = int(np.round(avg_height/8)*4)
        
        self.sliding_window_size = (height, width)
            
        return (height, width)
    
    
    def reduce_dimension(self, n_components=100, to_return=False):
        """ Aplikuje PCA a redukuje tim pocet priznaku """

        features = self.features        
        data = list()
        labels = list()
        
        # namapovani na numpy matice pro PCA
        for value in features.values():
            data.append(value["feature_vect"])
            labels.append(value["label"])
        
        X = np.vstack(data)
        Y = np.array(labels)        
        
        # PCA
        pca = PCA(n_components=self.feature_vector_length)   # vytvori PCA
        pca.fit(X, Y)
        reduced = pca.transform(X)      # redukuje dimenzi vektoru priznaku
        
        # znovu namapuje na zavedenou strukturu
        for i, feature_vect in enumerate(reduced):
            img_id = features.keys()[i]
            features[img_id]["feature_vect"] = list(feature_vect)
            features[img_id]["label"] = labels[i]
        
        # ulozeni PCA
        self.dataset.save_obj(pca, self.PCA_path+"/PCA_"+self.descriptor_type+".pkl")
        self.PCA_object = pca
        
        if to_return: return features
    
    
    def reduce_single_vector_dimension(self, vect):
        """ Nacte model PCA a aplikuje jej na jediny vektor """
            
        # nacteni jiz vypocteneho PCA, pokud jeste neni nectene
        if self.PCA_object is None:
            self.PCA_object = self.dataset.load_obj(self.PCA_path+"/PCA_"+self.descriptor_type+".pkl")
        
        # aplikace ulozeneho PCA
        reduced = self.PCA_object.transform(vect)      # redukuje dimenzi vektoru priznaku
        
        return reduced


class HOG(Extractor):
    
    def __init__(self, configpath="configuration/", configname="CT.json", orientations=12, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        
        super(HOG, self).__init__(configpath, configname)
        
        self.descriptor_type = 'hog'        
        
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
    
    # TODO: vymazat z parametru ten gaussian
    def skimHOG(self, gray, gaussian=False):
        """ Vrati vektor HOG priznaku """

        hist = hog(gray, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell,
                            cells_per_block=self.cells_per_block)  # hog je 1 dlouhy vektor priznaku, nesmi tam byt to visualize
        hist[hist<0] = 0
        
        return hist

        
    def extract_single_feature_vect(self, gray):
        """ Vrati vektor priznaku pro jedek obrazek """
        
        hist = self.skimHOG(gray)
        reduced = self.reduce_single_vector_dimension(hist)
        
        return reduced


    def extract_features(self, to_save=False, multiple_rois=None):
        """ Spocte vektory HOG priznaku pro trenovaci data a pro negatives ->
            -> pote je olabeluje 1/-1 a ulozi jako slovnik do .json souboru """
            
        features = self.features
        # s augmentaci nebo bez
        if multiple_rois is None: 
            multiple_rois = bool(self.dataset.config["data_augmentation"])
        
        print "Nacitaji se Trenovaci data ...",
        
        # Trenovaci data - obsahujici objekty
        for imgname in self.dataset.orig_images:
            
            if self.dataset.annotations.has_key(imgname):
                
                img = self.dataset.load_image(imgname)    # nccte obrazek
                boxes = self.dataset.annotations[imgname] # nacte bounding box
                
                for b, box in enumerate(boxes):
                
                    roi = self.get_roi(img, box, new_size = tuple(self.sliding_window_size), 
                                       image_processing=self.image_preprocessing)                            # vytahne region z obrazu
                    # augmentace dat
                    rois = self.multiple_rois_generator([roi]) if multiple_rois else [roi]                   # ruzne varianty roi
                    
                    # smycka, kdybych chtel ulozit roi v ruznych natocenich napriklad
                    for i, roi in enumerate(rois):
                        # extrahuje vektory priznaku regionu
                        features_vect = self.skimHOG(roi)
                        
                        # ulozi se do datasetu
                        img_id = imgname+"_"+str(b)+"_"+str(i)
                        features[img_id] = dict()
                        features[img_id]["label"] = 1
                        features[img_id]["feature_vect"] = list(features_vect)
                        
                        # pripadne ulozeni okenka
                        if to_save:
                            self.dataset.save_obj(roi, self.dataset.config["frames_positives_path"]+os.path.basename(img_id.replace(".pklz",""))+".pklz")

        print "Hotovo"
        print "Nacitaji se Negativni data ...",
        
        # Negativni data - neobsahujici objekty
        negatives = self.dataset.negatives
        for i in xrange(self.dataset.config["number_of_negatives"]):
            
            # nahodne vybere nejake negativni snimky
            gray = self.dataset.load_image(random.choice(negatives))
            rois = extract_patches_2d(gray, tuple(self.sliding_window_size), max_patches = self.dataset.config["number_of_negative_patches"])
            # predzpracovani obrazu
            if self.image_preprocessing: rois = self.image_processing(rois)      # intenzitni transformace
            # augmentace dat
            rois = self.multiple_rois_generator(rois) if multiple_rois else rois
            
            for j, roi in enumerate(rois):
                # extrakce vektoru priznaku
                features_vect = self.skimHOG(roi)
                
                # ulozeni do trenovaci mnoziny
                img_id = "negative_"+str(i)+"-"+str(j)
                features[img_id] = dict()
                features[img_id]["label"] = -1
                features[img_id]["feature_vect"] = list(features_vect)
                # pripadne ulozeni okenka
                if to_save:
                    self.dataset.save_obj(roi, self.dataset.config["frames_negatives_path"]+os.path.basename(img_id)+".pklz")
        
        # redukce dimenzionality
        features = self.reduce_dimension(to_return=True)      # pouzije PCA
        
        print "Hotovo"
        print "Probiha zapis trenovacich dat do souboru", 
        print self.dataset.config["training_data_path"]+"hog_features.json ...",

        # trenovaci data se zapisou se do jsonu
        self.dataset.zapis_json(features, self.dataset.config["training_data_path"]+"hog_features.json")
        
        print "Hotovo"
        
        return features

# TODO: problem, ze vetsinou nic nenalezne v rezech
class Others(Extractor):
    """ SIFT, SURF, ORB """
    
    def __init__(self, configpath = "configuration/", configname = "CT.json"):
        
        super(Others, self).__init__(configpath, configname)
        
        self.descriptor_type = str()

   
    def extract_single_feature_vect(self, gray):
        """ Vrati vektor priznaku pro jedek obrazek """
        
        gray = gray.astype('uint8')# Nutno pretypovat na int

        feature_detector = cv2.FeatureDetector_create(self.descriptor_type)
        extractor = cv2.DescriptorExtractor_create(self.descriptor_type)
        
        descriptor_list = list()
        
        keypoints = feature_detector.detect(gray)
        keypoints, descriptor = extractor.compute(gray, keypoints)
        
        descriptor = np.zeros((7, self.dataset.config[self.descriptor_type+"_descriptor_length"])) if descriptor is None else descriptor    # TODO: jen experiment
        descriptor_list.append(("test_data", descriptor.astype('float32')))
        
        # prvni deskriptor
        descriptors = descriptor_list[0][1]
        
        # provede k.means shlukovani
        k = 100
        voc, variance = kmeans(descriptors, k, 1)
        
        # spocita se histogram priznaku
        features_vects = np.zeros((len(descriptor_list), k)).astype(float)
        
        for i in xrange( len(descriptor_list) ):
            words, distance = vq(descriptor_list[i][1],voc)
            
            for word in words:
                features_vects[i][word] += 1
        
        return features_vects

    # TODO: mozna pridat do negativnich taky nuly a ne nic, jak to delam ted    
    def extract_features(self, multiple_rois=False):
        """ Extrahuje vektory priznaku pro SIFT, SURF nebo ORB """
        features = self.features
        labels = list()
        
        feature_detector = cv2.FeatureDetector_create(self.descriptor_type)
        extractor = cv2.DescriptorExtractor_create(self.descriptor_type)
        
        descriptor_list = list()
        
        print "Nacitaji se Trenovaci data...",
        
        # Trenovaci data - obsahujici objekty
        for imgname in self.dataset.orig_images:
            
            if self.dataset.annotations.has_key(imgname):
                
                img = self.dataset.load_image(imgname)     # nacte obrazek
                boxes = self.dataset.annotations[imgname]  # nacte bounding boxy pro tento obrazek

                for b, box in enumerate(boxes):
                    roi = self.get_roi(img, box, new_size = tuple(self.sliding_window_size)).astype('uint8')    # vytahne region z obrazu
                    rois = self.multiple_rois_generator([roi]) if multiple_rois else [roi]                           # kdybychom chteli otacet atd.
                    
                    for i, roi in enumerate(rois):
                        
                        keypoints = feature_detector.detect(roi)
                        keypoints, descriptor = extractor.compute(roi, keypoints)
                        descriptor = np.zeros((7, self.dataset.config[self.descriptor_type+"_descriptor_length"])) if descriptor is None else descriptor    # TODO: jen experiment
                        descriptor_list.append((imgname+"_"+str(b)+"_"+str(i), descriptor.astype('float32'))) # descriptory maji stejne delky, ale je jich ruzny pocet matice N x 158 napr.
                        labels.append(1)

        print "Hotovo"
        print "Nacitaji se Negativni data ...",
            
        # Negativni data - neobsahujici objekty
        negatives = self.dataset.negatives
        for i in xrange(self.dataset.config["number_of_negatives"]):
            
            # nahodne vybere nejake negativni snimky
            #img = cv2.imread(random.choice(negatives))
            gray = self.dataset.load_image(random.choice(negatives))
            rois = extract_patches_2d(gray, tuple(self.sliding_window_size), max_patches = self.dataset.config["number_of_negative_patches"])
            rois = self.multiple_rois_generator(rois) if multiple_rois else rois          # ruzne varianty
            
            for j, roi in enumerate(rois):
                
                roi = roi.astype('uint8')
                keypoints = feature_detector.detect(roi)
                keypoints, descriptor = extractor.compute(roi, keypoints)
                
                if not descriptor is None:
                    descriptor_list.append(("negative_"+str(i)+"-"+str(j), descriptor.astype('float32'))) # descriptory maji stejne delky, ale je jich ruzny pocet matice N x 158 napr.
                    labels.append(-1)
            
        # prvni deskriptor, pak uz jen pripina dalsi
        descriptors = descriptor_list[0][1]
        for image_path, descriptor in descriptor_list[1:]:
                descriptors = np.vstack((descriptors, descriptor))
        
        # provede k.means shlukovani
        k = 100
        voc, variance = kmeans(descriptors, k, 1)
        
        # spocita se histogram priznaku
        features_vects = np.zeros((len(descriptor_list), k)).astype(float)
        
        for i in xrange( len(descriptor_list) ):
            words, distance = vq(descriptor_list[i][1],voc)
            
            for word in words:
                features_vects[i][word] += 1
        
        labels = np.array(labels)
        
        features = self.features
                
        for i in xrange(len(descriptor_list)):
            
            img_id = descriptor_list[i][0]
            
            features[img_id] = dict()
            features[img_id]["label"] = labels[i]
            features[img_id]["feature_vect"] = list(features_vects[i])
            
        print "Hotovo"
        print "Probiha zapis trenovacich dat do souboru",
        print self.dataset.config["training_data_path"]+self.descriptor_type+"_features.json ...",

        # trenovaci data se zapisou do .json formatu
        self.dataset.zapis_json(features, self.dataset.config["training_data_path"]+self.descriptor_type+"_features.json")
        
        print "Hotovo"
        
        return features


class SIFT(Others):
    
    def __init__(self, configpath = "configuration/", configname = "CT.json"):
        
        super(SIFT, self).__init__(configpath, configname)
        
        self.config_path = configpath + configname
        self.config = self.dataset.precti_json(configpath + configname)
        
        self.descriptor_type = 'SIFT'


class SURF(Others):
    
    def __init__(self, configpath = "configuration/", configname = "CT.json"):
        
        super(SURF, self).__init__(configpath, configname)
        
        self.config_path = configpath + configname
        self.config = self.dataset.precti_json(configpath + configname)
        
        self.descriptor_type = 'SURF'


class ORB(Others):
    
    def __init__(self, configpath = "configuration/", configname = "CT.json"):
        
        super(ORB, self).__init__(configpath, configname)
        
        self.config_path = configpath + configname
        self.config = self.dataset.precti_json(configpath + configname)
        
        self.descriptor_type = 'ORB'




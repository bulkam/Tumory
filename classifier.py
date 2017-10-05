# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 18:54:05 2017

@author: mira
"""

import numpy as np
import cv2

import copy
import time
import re
import datetime as dt

from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import cPickle

import viewer
import feature_extractor as fe
import file_manager as fm


class Classifier():
    
    def __init__(self, configpath="configuration/", configname="CT.json", extractor=fe.HOG() , C=0.01):
        
        self.config_path = configpath + configname
        
        self.dataset = extractor.dataset#data_reader.DATAset(configpath, configname)
        
        self.extractor = extractor
        self.descriptor_type = self.extractor.descriptor_type
        
        self.config = self.dataset.config
        self.C = self.config["C"]  
        
        self.training_data_path = self.config["training_data_path"]+self.descriptor_type+"_features.json"
        self.classifier_path = self.config["classifier_path"]+"SVM-"+self.descriptor_type+".cpickle"
        
        self.data = None
        self.labels = None
        
        self.test_classifier = None
        self.test_results = dict()
        self.false_positives = dict()
        
        self.evaluation_modes = list()
        self.scores = list()
        self.evaluation_test_metrics = [accuracy_score,
                                        precision_score,
                                        recall_score,
                                        f1_score]
        
    
    
    def get_new_classifier(self):
        """ Vytvori a vrati instanci SVC """
        
        return SVC(kernel="linear", C = self.C, probability=True, random_state=42)
    
    
    def train(self):
        """ Natrenuje klasifikator a ulozi jej do souboru """
        
        # zalogovani zpravy
        self.dataset.log_info("[INFO] Trenuje se klasifikator ")
        
        data = self.data
        labels = self.labels
        
        print "[INFO] Trenuje se klasifikator... ",
        classifier = SVC(kernel="linear", C = self.C, probability=True, random_state=42)
        classifier.fit(data, labels)
        print "Hotovo"
        
        # ulozi klasifikator do .cpickle souboru
        print "[INFO] Uklada se klasifikator do souboru .cpickle ...",
        f = open(self.classifier_path, "w")
        f.write(cPickle.dumps(classifier))
        f.close()
        print "Hotovo"
        
        # zalogovani zpravy   
        self.dataset.log_info("... Hotovo.")
    
    
    def load_classifier(self):
        """ Nacte natrenovany klasifikator ze souboru """
        
        return cPickle.loads( open( self.config["classifier_path"]+"SVM-"+self.descriptor_type+".cpickle" ).read() )
        
        
    def create_training_data(self, mode="train", features=None):
        """ Vytvori trenovaci data a labely """
        
        print "[INFO] Nacitam trenovaci data... ", 
        
        TM = self.dataset.precti_json(self.config["training_data_path"]+self.descriptor_type+"_features.json" )
        if mode == "test" or not features is None:
            TM = features
            
        data = list()
        labels = list()
        
        for value in TM.values():
            
            data.append(value["feature_vect"])
            labels.append(value["label"])
        
        self.data = np.vstack(data)
        self.labels = np.array(labels)

        print "Hotovo"
    
    
    def store_false_positives(self):
        """ Ulozi feature vektory false positivu do trenovaci mnoziny """
        
        print "[INFO] Ukladani false positives mezi trenovaci data... ",
        TM = self.dataset.precti_json(self.config["training_data_path"]+self.descriptor_type+"_features.json")
        # pridavani false positives vektoru
        TM.update(self.false_positives)
        # ulozeni trenovacich dat
        self.dataset.zapis_json(TM, self.config["training_data_path"]+self.descriptor_type+"_features.json")
        # aktualizace trenovaci mnoziny
        self.create_training_data()
        
        print "Hotovo"
    
    
    def store_results(self):
        """ Ulozi vysledky testovani """

        # file manager
        manager = fm.Manager()
        
        # vytvoreni slozky 
        t = time.time()
        tstamp = str(dt.datetime.fromtimestamp(t))
        tstamp = re.sub(r'\s', '__', tstamp)
        tstamp = re.sub(r'[\:\.]', '-', tstamp)
    
        foldername = self.config["result_path"] + tstamp + "/"
        manager.make_folder(foldername)
        
        save_json = self.dataset.zapis_json
        
        # ulozeni konfigurace
        save_json(self.dataset.config, foldername+"CT.json")
        fm.copyfile(self.config_path, foldername+"CT-copyfile.json")
        # ulozeni konfigurace pri extrakci dat
        fm.copyfile("CTs/Configuration/config.json", foldername+"CTs-config.json")
        # TODO: ulozeni dalsich specialnich nastaveni a poznamek
        save_json(dict(), foldername+"notes.json")
        # ulozeni logovacho souboru
        fm.copyfile(self.dataset.config["log_file_path"], foldername+"LOG.log")
        # ulozeni PCA objektu
        fm.copyfile(self.config["PCA_path"]+"/PCA_"+self.extractor.descriptor_type+".pkl",
                    foldername+"/PCA_"+self.extractor.descriptor_type+".pkl")
        # ulozeni obrazku vysledku
        fm.copytree(self.dataset.config["results_PNG_path"], foldername+"PNG_results")
        # ulozeni vysledku
        save_json(self.test_results, foldername+"test_results.json")
        # TODO: ulozeni ohodnoceni vysledku - vybrat vsechny soubory te slozky 
        #       a ulozit je do hlavni slozky s vysledky
        for mode in self.evaluation_modes:
            fm.copyfile(self.config["evaluation_path"]+mode+"_evaluation.json", 
                        foldername+"evaluation_"+mode+".json")
        
        # ulozeni seznamu trenovacich obrazku
        images = {"positives": self.dataset.orig_images,
                  "negatives": self.dataset.negatives,
                  "test_images": self.dataset.test_images}
        save_json(images, foldername+"images_list.json")
        
        # ulozeni bounding boxu
        fm.copyfile(self.dataset.annotations_path, foldername+"boxes.json")
        # ulozeni trenovacich dat
        fm.copyfile(self.training_data_path, 
                    foldername+self.descriptor_type+"_features.json")
        
        # nakonec ulozeni modelu
        svm_filename = "SVM-"+self.descriptor_type+".cpickle"
        fm.copyfile(self.classifier_path, foldername+svm_filename)
            
    
    def classify_frame(self, gray, imgname):
        """ Pro dany obraz extrahuje vektor priznaku a klasifikuje jej """
        
        # extrakce vektoru priznaku
        roi = cv2.resize(gray, tuple(self.extractor.sliding_window_size), interpolation=cv2.INTER_AREA)
        feature_vect = self.extractor.extract_single_feature_vect(roi)
        
        # klasifikace pomoci testovaneho klasifikatoru
        result = list([np.array([self.test_classifier.predict_proba(feature_vect)[0, 1]])])    # klasifikace obrazu
        #result = list([self.test_classifier.predict(feature_vect)])    # klasifikace obrazu
        
        return result


    def classify_image(self, gray, mask, imgname, visualization=False, 
                       final_visualization=False, HNM=False, to_print=False):
        """ Pro dany obraz provede: 
        Projede vstupni obrazek pomoci sliding window a zmen mmeritka.
        Kazdy frame klasifikuje a ulozi vysledek.
        V pripade HNM zjisti u kazdeho framu, zda se jedna o false positive
            a pokud ano ulozi jej jak do obrazku, tak i do trenovacich dat
            (features).
        Nakonec se vysledky ulozi do .json souboru.
        
        Arguments:
            gray -- vstupni obrazek, ktery chceme klasifikovat
            mask -- maska pro vstupni obrazek
            imgname -- jmeno obrazku
            visualization -- zda ma byt zobrazeny prubeh testovani na obrazku
            final_visualization -- zda ma byt vykreseln vysledek na konci
            HNM -- zda jde o Hard negative mining
        """
        
        # ve vysledcich se zalozi polozka s timto obrazkem a tam budu pridavat vysledky pro jednotlive framy
        self.test_results[imgname] = list()
        # false positives
        false_positives = self.false_positives
        # abych mel prehled kolik framu to detekuje
        n_detected = 0
        # skutecne klasifikovane
        n_positive_bounding_boxes = 0
        
        # nacteni window_size a dalsich parametru z konfigurace
        window_size = self.config["sliding_window_size"]
        pyramid_scale = self.config["pyramid_scale"]
        sliding_window_step = self.config["sliding_window_step"]
        image_preprocessing = bool(self.config["image_preprocessing"])
        
        # minimalni pravdepodobnost framu pro detekci
        min_prob = self.config["min_prob"]
        # minimalni nutne zastoupeni jater ve framu
        min_liver_coverage = self.config["min_liver_coverage"]
        min_liver_center_coverage = self.config["min_liver_center_coverage"]
        liver_center_coverage_mode = bool(self.config["liver_center_coverage_mode"])
        ellipse_mode = bool(self.config["ellipse_mode"])
        liver_sides_mode = bool(self.config["liver_sides_mode"])
        min_liver_sides_filled = bool(self.config["min_liver_sides_filled"])
        min_liver_side_coverage = bool(self.config["min_liver_side_coverage"])
        # minimalni nutne zastoupeni artefaktu ve framu - pro HNM
        min_HNM_coverage = self.config["min_HNM_coverage"]
        
        for scaled in self.extractor.pyramid_generator(gray, scale=pyramid_scale):
            
            # spocteni meritka
            scale = float(gray.shape[0])/scaled.shape[0]
            
            for bounding_box, frame in self.extractor.sliding_window_generator(img = scaled, 
                                                                               step = sliding_window_step,
                                                                               window_size = window_size,
                                                                               image_processing=image_preprocessing):
                                                                                   
                # Pokud se tam sliding window uz nevejde, prejdeme na dalsi                
                if frame.shape != tuple(window_size):
                    continue
                
                # klasifikace obrazu
                result = self.classify_frame(frame, imgname)
                
                # spocteni bounding boxu v puvodnim obrazku beze zmeny meritka
                real_bounding_box = (x, h, y, w) = list( ( scale * np.array(bounding_box) ).astype(int) )   
                # zjisteni, zda se ot nachczi v jatrech
                mask_frame = fe.get_mask_frame(mask, real_bounding_box)
                frame_liver_coverage = fe.liver_coverage(mask_frame)
                frame_liver_center_coverage, real_mini_bounding_box = fe.liver_center_coverage(mask_frame, real_bounding_box)
                frame_liver_center_ellipse_coverage, small_mask = fe.liver_center_ellipse_coverage(mask_frame)
                
                # ulozeni vysledku
                image_result = { "scale": scale,
                                 "bounding_box": real_bounding_box,
                                 "result": list(result[0]),
                                 "liver_coverage": frame_liver_coverage,
                                 "liver_center_coverage": frame_liver_center_coverage,
                                 "liver_center_ellipse_coverage": frame_liver_center_ellipse_coverage}
                self.test_results[imgname].append(image_result)
                
                # upozorneni na pozitivni data
                if result[0] > min_prob and to_print and not HNM:
                    print "[RESULT] Nalezen artefakt: ", image_result, frame_liver_coverage
                    n_detected += 1
                
                # podminka detekce
                detection_condition = (result[0] > min_prob) and (frame_liver_coverage >= min_liver_coverage)
                # pokud nas zajima zastoupeni jater ve stredu
                if liver_center_coverage_mode:
                    if ellipse_mode:
                        detection_condition = (result[0] > min_prob) and (frame_liver_center_ellipse_coverage >= min_liver_center_coverage)
                        real_mini_bounding_box = None
                    else:
                        detection_condition = (result[0] > min_prob) and (frame_liver_center_coverage >= min_liver_center_coverage)
                        small_mask = None
                else:
                    real_mini_bounding_box = None
                    small_mask = None
                
                if liver_sides_mode:
                    sides_coverage, sides_filled = fe.liver_sides_filled(mask_frame, min_coverage=min_liver_side_coverage)
                    detection_condition = detection_condition and (sides_filled >= min_liver_sides_filled)
                
                # oznaceni jako pozitivni nebo negativni
                self.test_results[imgname][-1]["mark"] = int(detection_condition)
                # pocitani pozitivnich bounding boxu
                n_positive_bounding_boxes += self.test_results[imgname][-1]["mark"]
                
                # pripadne Hard Negative Mining
                if detection_condition and HNM:
                    # zjisteni skutecneho vyskytu atrefaktu
                    frame_artefact_coverage = fe.artefact_coverage(mask_frame)
                    if visualization: 
                        print "Artefact coverage: ", frame_artefact_coverage
                    # pokud je detekovan, ale nemel by byt
                    if frame_artefact_coverage <= min_HNM_coverage:
                        
                        img_id = "false_positive_"+imgname+"_scale="+str(scale)+"_bb="+str(x)+"-"+str(h)+"-"+str(y)+"-"+str(w)
                        print "[RESULT] False positive !!!"
                        # extrakce vektoru priznaku
                        result_roi = cv2.resize(frame, tuple(self.extractor.sliding_window_size), interpolation=cv2.INTER_AREA)
                        result_feature_vect = list(self.extractor.extract_single_feature_vect(result_roi)[0])
                        # ulozeni do false positives
                        false_positives[img_id] = {"feature_vect":result_feature_vect, "label":-1}
                        # pripadne ukladani framu
                        img_id_to_save = "false_positive_"+fm.get_imagename(imgname)+"_bb="+str(x)+"-"+str(h)+"-"+str(y)+"-"+str(w)
                        # ulozeni mezi fp obrazky
                        if bool(self.config["FP_to_FP"]):
                            self.dataset.save_image(frame, self.config["false_positives_path"]+img_id_to_save+".png")
                            self.dataset.save_image(frame, self.config["false_positives_path"]+img_id_to_save+".pklz")
                        # ulozeni mezi negatives
                        if bool(self.config["FP_to_negatives"]):
                            self.dataset.save_image(frame, self.config["negatives_path"]+img_id_to_save+".pklz")
                        # ulozeni mezi framy
                        if bool(self.config["FP_to_frames"]):
                            self.dataset.save_image(fe.get_mask_frame(gray, real_bounding_box),
                                                    self.config["frames_HNM_path"]+img_id_to_save+".png")
                            self.dataset.save_image(fe.get_mask_frame(gray, real_bounding_box),
                                                    self.config["frames_HNM_path"]+img_id_to_save+".pklz")
                            
                            
                # pripadna vizualizace projizdeni slidong window
                if visualization:
                    viewer.show_frame_in_image(gray, real_bounding_box, 
                                               small_box=real_mini_bounding_box,
                                               small_mask=small_mask,
                                               detection=detection_condition, 
                                               blured=True, sigma=5, 
                                               mask=mask)
                
        # non-maxima suppression 
        if n_positive_bounding_boxes >= 1:
            print n_positive_bounding_boxes
            detected_boxes = self.non_maxima_suppression(imgname)
        else:
            detected_boxes = list()
        
        # pripadna vizualizace
        if final_visualization:
            viewer.show_frames_in_image(copy.copy(gray), self.test_results[imgname], 
                                        save_path=self.config["results_PNG_path"],
                                        fname=fm.get_imagename(imgname))
            viewer.show_frames_in_image_nms(copy.copy(gray), 
                                            detected_boxes,
                                            mask=copy.copy(mask),
                                            save_path=self.config["results_PNG_path"],
                                            fname=fm.get_imagename(imgname))
        
        if HNM:
            print "[RESULT] Celkem nalezeno ", len(false_positives), "false positives."

        
        print "[RESULT] Celkem nalezeno ", n_detected, " artefaktu."
        print "[RESULT] ", n_positive_bounding_boxes, " bounding boxu nakonec vyhodnoceno jako pozitivni."


    def classify_test_images(self, visualization=False, 
                             final_visualization=False,
                             to_print=False):
        """ Nacte testovaci data a klasifikuje je """
        
        # zalogovani zpravy
        self.dataset.log_info("[INFO] Klasifikuji se snimky... ")
        
        # nacteni testovaneho klasifikatoru
        self.test_classifier = self.load_classifier()
        
        imgnames = self.dataset.test_images
        
        for i, imgname in enumerate(imgnames[1:2]): # 1:2
            
            print "[INFO] Testovani obrazku "+imgname+" ("+str(i)+".)..."
            # nacteni obrazu
            gray = self.dataset.load_image(imgname)
            # nacteni masky
            maskname = re.sub("test_images", "masks", imgname)
            mask = self.dataset.load_image(maskname)
            
            # klasifikace obrazu
            self.classify_image(gray, mask, imgname, 
                                visualization=visualization,
                                final_visualization=final_visualization,
                                to_print=to_print)
        
        # ulozeni do souboru vysledku
        self.dataset.zapis_json(self.test_results, self.config["test_results_path"])
        
        # zalogovani zpravy   
        self.dataset.log_info("      ... Hotovo.")
    
    # TODO: zkouset
    def hard_negative_mining(self, visualization=False, 
                             final_visualization=False,
                             origs=[0, 0], HNMs=[0, 0]):
        """ Znovu projede tranovaci data a false positives ulozi do negatives """
        
        # zalogovani zpravy
        self.dataset.log_info("[INFO] Hard Negative Mining ")
        self.dataset.log_info("       positives: "+str(origs)+"  | HNMs: "+str(HNMs))
        # nacteni testovaneho klasifikatoru
        self.test_classifier = self.load_classifier()  
        
        # HNM na pozitivnich rezech
        imgnames = self.dataset.orig_images
        
        for i, imgname in enumerate(imgnames[origs[0]:origs[1]]): #20-30
            
            if not "=" in imgname: # tetsovani jen originalnich dat
            
                print "[INFO] Testovani obrazku "+imgname+" ("+str(i)+".P)..."
                # nacteni obrazu
                gray = self.dataset.load_image(imgname)
                # nacteni masky
                maskname = re.sub("orig_images", "masks", imgname)
                mask = self.dataset.load_image(maskname)
                
                # klasifikace obrazu
                self.classify_image(gray, mask, imgname, HNM=True, 
                                    visualization=visualization,
                                    final_visualization=final_visualization)
        
        # ted na negativech
        imgnames = self.dataset.HNM_images
        
        for i, imgname in enumerate(imgnames[HNMs[0]:HNMs[1]]): # [40:41] # 30-60, 60-90, 90-150
            
            print "[INFO] Testovani obrazku "+imgname+" ("+str(i)+".HNM)..."
            # nacteni obrazu
            gray = self.dataset.load_image(imgname)
            # nacteni masky
            maskname = re.sub("hard_negative_mining", "masks", imgname)
            mask = self.dataset.load_image(maskname)
            
            # klasifikace obrazu
            self.classify_image(gray, mask, imgname, HNM=True, visualization=visualization)
        
        self.store_false_positives()
        # zalogovani zpravy
        self.dataset.log_info("      ... Hotovo.")
    
    
    def create_boxes_nms(self, imgname):
        """ Vybere pravi bounding boxy pro dany obrazek pro nms """
        
        results = self.test_results
        # pokud nebudou vysledky obsahovat klic, 
        # tak se zkusit podivat do souboru
        if not results.has_key(imgname):
            results = self.dataset.precti_json(self.dataset.config["test_results_path"])
            
        # inicializace boxu a jejich pravdepodobnosti
        boxes = list()
        probs = list()
        
        for result in results[imgname]:
            # ukladani jen tech pozitivnich
            if result["mark"] == 1:
                boxes.append(np.hstack(result["bounding_box"]))
                probs.append(result["result"][0])

        boxes = np.vstack(boxes).astype(float)
        probs = np.hstack(probs)
    
        return boxes, probs
    
    # TODO: zkouset optimalni prah
    def non_maxima_suppression(self, imgname, overlap_thr=0.01):
        """ Provede redukci prekryvajicich se bounding boxu """
        
        overlap_thr = self.dataset.config["NMS_overlap_thr"]
        # nacteni pozitivnich bounding boxu a jejich ppsti
        boxes, probs = self.create_boxes_nms(imgname)
        # nacteni jednotlivych souradnic bounding boxu
        ys, hs, xs, ws = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        bb_area = (hs-ys+1)*(ws-xs+1)
        # serazeni indexu podle pravdepodobnosti
        indexes = np.argsort(probs)
        new_indexes = list()
        
        while True:
            # vytazeni indexu bounding boxu s nejvyssi ppsti
            i = indexes[-1]
            new_indexes.append(i)
            index = indexes[:-1]
            
            ys2 = np.maximum(ys[i], ys[index])
            hs2 = np.minimum(hs[i], hs[index])
            xs2 = np.maximum(xs[i], xs[index])
            ws2 = np.minimum(ws[i], ws[index])
            
            new_area = np.maximum(0, ws2-xs2+1) * np.maximum(0, hs2-ys2+1)
            # vypocet prekryti
            overlap = new_area.astype(float) / bb_area[index]
            # odstraneni indexu boxu s vysokym prekrytim
            indexes_to_delete = np.where(overlap > overlap_thr)[0]
            indexes = np.delete(index, indexes_to_delete)
            
            # zastavit po vyprazdneni
            if len(indexes) == 0:
                break

        print "[RESULT] Pocet pozitivnich bounding boxu zredukovan na ", len(new_indexes)
        #print new_indexes
        # vrati vybrane bounding boxy
        return boxes[new_indexes].astype(int)
    
    # TODO: okomentovat
    def evaluate(self, mode='test', cv_scorings=['accuracy'], to_train=False,
                 method="CV"):
        """ Ohodnoti klasifikator podle zvoleneho kriteria """
        
        self.extractor.features = {}
        
        if mode == "test":
            
            self.dataset.orig_images, self.dataset.negatives = [], []
            positives, negatives = self.get_test_data()
            #print positives
            #print negatives
            
            print len(positives), "pozitivnich a ", len(negatives), " negativnich"
            
            self.extractor.n_negatives = len(negatives) + 1
            self.extractor.n_negative_patches = 2
        
        # jinak se refreshuje dataset
        elif mode == "train":
            self.dataset.create_dataset_CT()
        
        # nastaveni modu pro extrakci features
        extractor_mode = "transform" if mode == "test" else "normal"
        # rovnou volame vects, abychom mohli nastavit transform mode
        self.extractor.extract_feature_vects(multiple_rois=False, 
                                             save_features=False,
                                             mode=extractor_mode)
                                             
        self.create_training_data(mode=mode, features=self.extractor.features)
        X, y = self.data, self.labels
        # pripadne trenovani
        if to_train and mode == "test":
            self.train()
        
        print "Celkem dat: "
        print "   " + str( len([s for s in y if s > 0]) ) + " pozitivnich"
        print "   " + str( len([s for s in y if s < 0]) ) + " negativnich"
               
        # ohodnoceni
        scores = dict()
        # pro trenovaci data delame cross validation
        if mode == "train":
            scores = cross_validate(self.get_new_classifier(),  # vytvori novy klasifikator
                                    X, y,
                                    scoring=cv_scorings)
            for key in scores.keys():
                scores[key] = list(scores[key])
            print "[RESULT] Vysledne skore: ", scores
            
        # pro testovaci data jen spocitame skore na jiz natrenovanem klasifikatoru
        elif mode == "test":
            # nacte se testovaci klasifikator, pokud jeste neni
            if self.test_classifier is None:
                self.test_classifier = self.load_classifier()
            # vytvoreni porovnavacich dat
            y_pred = self.test_classifier.predict(X)
            # pocitani skore
            for metric_method in self.evaluation_test_metrics:
                scores[metric_method.__name__] = list(metric_method(y_pred, y))
                print "[RESULT] Vysledne skore podle "+metric_method.__name__+": ", 
                print scores[metric_method.__name__]
        
        # zapsani vysledku
        self.scores.append(scores)
        # ulozeni vysledku ohodnoceni
        self.dataset.zapis_json(scores, self.config["evaluation_path"]+mode+"_evaluation.json")
        # ulozeni do seznamu typu provedenych ohodnoceni
        self.evaluation_modes.append(mode)
        
    # TODO: okomentovat, presunout
    def get_test_data(self):
        """ Extrahuje testovaci data rozdelena na positives a negatives """
        
        for imgname in self.dataset.test_images:
            orig_imgname = fm.get_orig_imgname(imgname)
            
            if self.dataset.annotations.has_key(orig_imgname):
                self.dataset.orig_images.append(imgname)
                self.dataset.annotations[imgname] = self.dataset.annotations[orig_imgname]
                
            else:
                self.dataset.negatives.append(imgname)
            
        return self.dataset.orig_images, self.dataset.negatives



            
        

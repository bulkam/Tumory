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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import cPickle

import viewer
import feature_extractor as fe
import file_manager as fm


class Classifier():
    
    def __init__(self, configpath="configuration/", configname="CT.json", extractor=fe.HOG() , C=0.01):
        
        self.config_path = configpath + configname
        
        self.dataset = extractor.dataset#data_reader.DATAset(configpath, configname)
        
        self.extractor = extractor
        self.test_extractor = copy.copy(extractor)
        self.descriptor_type = self.extractor.descriptor_type
        
        self.config = self.dataset.config
        self.C = self.config["C"] 
        
        self.pyramid_scale = self.config["pyramid_scale"]
        self.sliding_window_step = self.config["sliding_window_step"]
        
        self.training_data_path = self.config["training_data_path"]+self.descriptor_type+"_features.json"
        self.classifier_path = self.config["classifier_path"]+"SVM-"+self.descriptor_type+".cpickle"
        
        self.data = None
        self.labels = None
        
        self.test_classifier = None
        self.test_results = dict()
        self.test_results_nms = dict()
        self.false_positives = dict()
        
        self.evaluation_modes = list()
        self.scores = list()
        self.evaluation_test_metrics = [accuracy_score,
                                        precision_score,
                                        recall_score,
                                        f1_score]
        
        # minimalni pravdepodobnost framu pro detekci
        self.min_prob = self.config["min_prob"]
        # minimalni nutne zastoupeni jater ve framu
        self.liver_coverage_mode = bool(self.config["liver_coverage_mode"])
        self.min_liver_coverage = self.config["min_liver_coverage"]
        self.min_liver_center_coverage = self.config["min_liver_center_coverage"]
        self.liver_center_coverage_mode = bool(self.config["liver_center_coverage_mode"])
        self.ellipse_mode = bool(self.config["ellipse_mode"])
        self.liver_sides_mode = bool(self.config["liver_sides_mode"])
        self.min_liver_sides_filled = bool(self.config["min_liver_sides_filled"])
        self.min_liver_side_coverage = bool(self.config["min_liver_side_coverage"])
        # minimalni nutne zastoupeni artefaktu ve framu - pro HNM
        self.min_HNM_coverage = self.config["min_HNM_coverage"]
        self.HNM_min_prob = self.config["HNM_min_prob"]
        
        # HNM nastaveni
        self.n_best_hnms = self.config["n_best_hnms"]
        self.double_HNM = bool(self.config["double_HNM"])
        self.HNM_edges_only = self.config["HNM_edges_only"]
        self.HNM_pyramid_scale = self.config["HNM_pyramid_scale"]
        self.HNM_sliding_window_step = self.config["HNM_sliding_window_step"]

    
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
        
        TM = {}
        if mode == "test" or not features is None:
            TM = features
        else:
            TM = self.dataset.precti_json(self.config["training_data_path"]+self.descriptor_type+"_features.json" )
            
        data = list()
        labels = list()
        
        for value in TM.values():
            
            data.append(value["feature_vect"])
            labels.append(value["label"])
        
        self.data = np.vstack(data)
        self.labels = np.array(labels)

        print "Hotovo"
    

    def get_test_data(self):
        """ Extrahuje testovaci data rozdelena na positives a negatives """
        
        print "[INFO] Rozdeluji testovaci data mezi positives a negatives... "
        
        # projede testovaci obrazky - framy
        for imgname in self.dataset.evaluation_test_images:
            orig_imgname = fm.get_orig_imgname(re.sub("\#+\d+", "", imgname))
            #print orig_imgname
            # rozdeli obrazky do positives a negatives
            if self.dataset.annotations.has_key(orig_imgname):
                # nesmi to byt kopie z trenovaci mnoziny
                if not orig_imgname.startswith("00_copy"):
                    self.dataset.orig_images.append(imgname)
                    self.dataset.annotations[imgname] = self.dataset.annotations[orig_imgname]
            else:
                self.dataset.negatives.append(imgname)
                
        print "Hotovo"
        # vrati positives a negatives
        return self.dataset.orig_images, self.dataset.negatives    
        
    # TODO:
    def select_best_hnms_by_value(self):
        """ Seradi snimky pro hnm podle obsahu jater v nich """
        
        hnms = self.dataset.HNM_images
        # pocitani jaternich pixelu ve snimcich
        mask_volumes = []
        for name in hnms:
            mask = fm.get_mask(name, self.config)
            mask_volumes.append(np.sum((mask >= 1).astype(int)))
            
        indxs = np.argsort(mask_volumes)[::-1]
        sorted_hnms = list(np.array(hnms)[indxs])
        
        return sorted_hnms
    
    
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
    
    
    def store_results(self, suffix=""):
        """ Ulozi vysledky testovani """

        # file manager
        manager = fm.Manager()
        
        # vytvoreni slozky 
        t = time.time()
        tstamp = str(dt.datetime.fromtimestamp(t))
        tstamp = re.sub(r'\s', '__', tstamp)
        tstamp = re.sub(r'[\:\.]', '-', tstamp)
    
        foldername = self.config["result_path"] + tstamp + suffix + "/"
        manager.make_folder(foldername+"scripts/")
        
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
        save_json(self.test_results_nms, foldername+"result_nms.json")
        # TODO: ulozeni ohodnoceni vysledku - vybrat vsechny soubory te slozky 
        #       a ulozit je do hlavni slozky s vysledky
        for mode in self.evaluation_modes:
            fm.copyfile(self.config["evaluation_path"]+mode+"_evaluation.json", 
                        foldername+"evaluation_"+mode+".json")
        # ulozeni vysledku ohodnoceni prekryti artefaktu s bounding boxy
        fm.copyfile(self.config["evaluation_path"]+"nms_overlap_evaluation.json",
                    foldername+"nms_overlap_evaluation.json")

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
        
        # ulozeni vsech skriptu
        scripts = [name for name in fm.os.listdir(".") if name.endswith('.py')]
        for script in scripts:
            fm.copyfile(script, foldername+"scripts/"+script)
        
        
        
    def classify_frame(self, gray, mask_frame, imgname, visualization=False):
        """ Pro dany obraz extrahuje vektor priznaku a klasifikuje jej """
        
        roi = gray.copy()
        
        # obarveni pozadi
        if self.extractor.background_coloring:
            roi = self.extractor.apply_background_coloring(roi, mask_frame)
        
        # zmena velikosti obrazku
        roi = cv2.resize(roi, tuple(self.extractor.sliding_window_size),
                         interpolation=cv2.INTER_AREA)

        # image processing
        if self.extractor.image_preprocessing:
            roi = self.extractor.apply_image_processing(roi)
            
        # extrakce vektoru priznaku
        feature_vect = self.extractor.extract_single_feature_vect(roi)
        
        # klasifikace pomoci testovaneho klasifikatoru
        result = list([np.array([self.test_classifier.predict_proba(feature_vect)[0, 1]])])    # klasifikace obrazu
        #result = list([self.test_classifier.predict(feature_vect)])    # klasifikace obrazu
        #result = list([np.array([self.test_classifier.predict(feature_vect)])])
        if visualization:
            cv2.imshow('frame3', cv2.resize(roi, (256, 256), 
                                            interpolation=cv2.INTER_AREA))        
            cv2.waitKey(1)
        
        # jen diagnostika
#        if result[0] > self.HNM_min_prob:
#            fvs = [list(line) for line in self.data]
#            if list(feature_vect) in fvs:
#                print "Je to v datech"
        
#        if result[0] > self.min_prob:
#            l = str(len([i for i in fe.os.listdir(self.config["frames_positives_path"]) if i.endswith(".png")]))
#            #print self.config["frames_positives_path"]+"img"+l+".png"            
#            self.dataset.save_image(roi, self.config["frames_positives_path"]+"img"+l+".png")
#        else:
#            l = str(len([i for i in fe.os.listdir(self.config["frames_negatives_path"]) if i.endswith(".png")]))
#            self.dataset.save_image(roi, self.config["frames_negatives_path"]+"img"+l+".png")
            
        return result, feature_vect


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
        window_size = self.extractor.sliding_window_size
        pyramid_scale = self.pyramid_scale if not HNM else self.HNM_pyramid_scale
        sliding_window_step = self.sliding_window_step if not HNM else self.HNM_sliding_window_step
        image_preprocessing = self.extractor.image_preprocessing
        
        # nacteni prahu pro podminku detekce
        # minimalni pravdepodobnost framu pro detekci
        min_prob = self.min_prob if not HNM else self.HNM_min_prob
        # minimalni nutne zastoupeni jater ve framu
        liver_coverage_mode = self.liver_coverage_mode
        min_liver_coverage = self.min_liver_coverage
        # minimalni zastoupeni jater ve framuctredu framu
        liver_center_coverage_mode = self.liver_center_coverage_mode
        min_liver_center_coverage = self.min_liver_center_coverage
        # zdali ma byt stredovy frame ve tvaru elipsy
        ellipse_mode = self.ellipse_mode
        # zda ma byt zahrnuta podminka vyplnenych stran
        liver_sides_mode = self.liver_sides_mode
        # minimalni pocet vyplnenych stran jatry
        min_liver_sides_filled = self.min_liver_sides_filled
        # minimalni procento vypnenych okrajovych pixelu 
        min_liver_side_coverage = self.min_liver_side_coverage
        # minimalni nutne zastoupeni artefaktu ve framu - pro HNM
        min_HNM_coverage = self.min_HNM_coverage
        
        for scaled, mask_scaled in self.extractor.pyramid_generator(gray, mask, scale=pyramid_scale):
            
            # spocteni meritka
            scale = float(gray.shape[0])/scaled.shape[0]
            
            for bounding_box, frame, mask_frame_scaled in self.extractor.sliding_window_generator(img = scaled,
                                                                                                  mask = mask_scaled,
                                                                                                  step = sliding_window_step,
                                                                                                  window_size = window_size,
                                                                                                  image_processing=image_preprocessing):
                                                                                   
                # Pokud se tam sliding window uz nevejde, prejdeme na dalsi                
                if frame.shape != tuple(window_size):
                    continue
                
                # spocteni bounding boxu v puvodnim obrazku beze zmeny meritka
                real_bounding_box = (x, h, y, w) = list( ( scale * np.array(bounding_box) ).astype(int) )   
                # zjisteni, zda se oblast nachazi v jatrech
                # spocteni pokryti ruznych oblasti jatry
                mask_frame = fe.get_mask_frame(mask, real_bounding_box)
                frame_liver_coverage = fe.liver_coverage(mask_frame)
                frame_liver_center_coverage, real_mini_bounding_box = fe.liver_center_coverage(mask_frame, real_bounding_box)
                frame_liver_center_ellipse_coverage, small_mask = fe.liver_center_ellipse_coverage(mask_frame)                
                
                # pokud bude frame uplne mimo jatra, tak dal nic nepocitat
                # pro testovani nejlepsich konfiguraci rovnou zahazovat 
                #               framy, co nesplnuji prekryti
                # potom pro dalsi tetsovani to muzu nehcta na 0
                #               nebo jen frame_liver_coverage
                if frame_liver_coverage == 0:
                    continue
                elif self.liver_coverage_mode and frame_liver_coverage < self.min_liver_coverage:
                    continue
                elif self.liver_center_coverage_mode :
                    if self.ellipse_mode:
                        if frame_liver_center_ellipse_coverage < self.min_liver_center_coverage:
                            continue
                    elif frame_liver_center_coverage < self.min_liver_center_coverage:
                        continue
                
                # klasifikace obrazu
                result, result_feature_vect = self.classify_frame(frame, 
                                                                  mask_frame_scaled, 
                                                                  imgname, 
                                                                  visualization=visualization)
                R = result[0]
                # ulozeni vysledku
                image_result = { "scale": scale,
                                 "bounding_box": real_bounding_box,
                                 "result": list(R),
                                 "liver_coverage": frame_liver_coverage,
                                 "liver_center_coverage": frame_liver_center_coverage,
                                 "liver_center_ellipse_coverage": frame_liver_center_ellipse_coverage}
                self.test_results[imgname].append(image_result)
                
                # upozorneni na pozitivni data
                if R > min_prob and to_print and not HNM:
                    print "[RESULT] Nalezen artefakt: ", image_result, frame_liver_coverage
                    n_detected += 1
                
                
                #       ----- Podminka detekce  -----
                # rozhodnuti klasifikatoru
                detection_condition = R > min_prob
                # pokud nas zajima zastoupeni jater v celem bb
                if liver_coverage_mode:
                    detection_condition = detection_condition and (frame_liver_coverage >= min_liver_coverage)
                # pokud nas zajima zastoupeni jater ve stredu
                if liver_center_coverage_mode:
                    if ellipse_mode:
                        detection_condition = detection_condition and (frame_liver_center_ellipse_coverage >= min_liver_center_coverage)
                        real_mini_bounding_box = None
                    else:
                        detection_condition = detection_condition and (frame_liver_center_coverage >= min_liver_center_coverage)
                        small_mask = None
                else:
                    real_mini_bounding_box = None
                    small_mask = None
                # pripadna podminka pokryti po stranach
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
                        print "Coverage | Artefact: ", frame_artefact_coverage,
                        print "Liver: ", frame_liver_coverage
                    # pokud je detekovan, ale nemel by byt
                    if frame_artefact_coverage <= min_HNM_coverage:
                        
                        img_id = "false_positive_"+imgname+"_scale="+str(scale)+"_bb="+str(x)+"-"+str(h)+"-"+str(y)+"-"+str(w)
                        #print "[RESULT] False positive !!!"
                        print "FP" , 
                        # extrakce vektoru priznaku
                        #result_roi = cv2.resize(frame, tuple(self.extractor.sliding_window_size), interpolation=cv2.INTER_AREA)
                        #result_feature_vect = list(self.extractor.extract_single_feature_vect(result_roi)[0])
                        # ulozeni do false positives
                        false_positives[img_id] = {"feature_vect":list(result_feature_vect[0]), "label":-1}
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
        # ulozeni vysledku pro dany obrazek    
        if not HNM: 
            self.test_results_nms[imgname] = [list(box) for box in detected_boxes]
        
        # pripadna vizualizace
        if final_visualization:
            viewer.show_frames_in_image(copy.copy(gray), self.test_results[imgname], 
                                        save_path=self.config["results_PNG_path"],
                                        fname=fm.get_imagename(imgname))
        if final_visualization or not HNM:                                
            viewer.show_frames_in_image_nms(copy.copy(gray), 
                                            detected_boxes,
                                            mask=copy.copy(mask),
                                            save_path=self.config["results_PNG_path"],
                                            fname=fm.get_imagename(imgname),
                                            to_show=final_visualization)
        
        if HNM:
            print "[RESULT] Celkem nalezeno ", len(false_positives), "false positives."

        print "[RESULT] Celkem nalezeno ", n_detected, " artefaktu."
        print "[RESULT] ", n_positive_bounding_boxes, " bounding boxu nakonec vyhodnoceno jako pozitivni."
    
    # TODO:
    def classify_test_images(self, visualization=False, 
                             final_visualization=False,
                             to_print=False):
        """ Nacte testovaci data a klasifikuje je """
        
        # zalogovani zpravy
        self.dataset.log_info("[INFO] Klasifikuji se snimky... ")
        
        # nacteni testovaneho klasifikatoru
        self.test_classifier = self.load_classifier()
        
        imgnames = self.dataset.test_images
        
        for i, imgname in enumerate(imgnames[1:]): ## 7:14, 7:8, 1:2 # negativni je 41:42
            
            print "[INFO] Testovani obrazku "+imgname+" ("+str(i)+".)..."
            # nacteni obrazu
            gray = self.dataset.load_image(imgname)
            # nacteni masky
            maskname = re.sub("test_images", "masks", imgname)
            mask = self.dataset.load_image(maskname)
            
            # augmentovane obrazky jsou moc velke, tak se oriznou
            if "AFFINE" in imgname:
                gray, mask = fe.cut_image(gray, mask)
            
            # klasifikace obrazu
            self.classify_image(gray, mask, imgname, 
                                visualization=visualization,
                                final_visualization=final_visualization,
                                to_print=to_print)
        
        # ulozeni do souboru vysledku
        self.dataset.zapis_json(self.test_results, self.config["test_results_path"])
        self.dataset.zapis_json(self.test_results_nms, self.config["result_path"]+"results_nms.json")
        
        # zalogovani zpravy   
        self.dataset.log_info("      ... Hotovo.")
    

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
            
            if not "=" in imgname: # testovani jen originalnich dat
            
                print "[INFO] Testovani obrazku "+imgname+" ("+str(i)+".P)..."
                # nacteni obrazu
                gray = self.dataset.load_image(imgname)
                # nacteni masky
                maskname = re.sub("orig_images", "masks", imgname)
                mask = self.dataset.load_image(maskname)
                
                # augmentovane obrazky jsou moc velke, tak se oriznou
                if "AFFINE" in imgname:
                    gray, mask = fe.cut_image(gray, mask)
                
                # klasifikace obrazu
                self.classify_image(gray, mask, imgname, HNM=True, 
                                    visualization=visualization,
                                    final_visualization=final_visualization)
        
        # ted na negativech
        #imgnames = self.dataset.HNM_images
        imgnames = self.select_best_hnms_by_value()#[0: min(self.n_best_hnms, len(self.dataset.HNM_images))]
        
        for i, imgname in enumerate(imgnames): 
                                    #imgnames[HNMs[0]:HNMs[1]],  [40:41] # 30-60, 60-90, 90-150
            
            print "[INFO] Testovani obrazku "+imgname+" ("+str(i)+".HNM)..."
            # nacteni obrazu
            gray = self.dataset.load_image(imgname)
            # nacteni masky
            maskname = re.sub("hard_negative_mining", "masks", imgname)
            mask = self.dataset.load_image(maskname)
            # klasifikace obrazu
            self.classify_image(gray, mask, imgname, HNM=True, visualization=visualization)
            # pokud jsme dosahli daneho poctu HNM snimku:
            if i == self.n_best_hnms:
                # pokud nechceme jeste pretrenovat HNM, tak kones
                if not self.double_HNM:
                    break
                # jinak:
                else:
                    self.store_false_positives()
                    self.false_positives = dict()
                    self.create_training_data()
                    self.train()
                    self.data, self.labels = None, None
                    self.HNM_min_prob = self.min_prob #self.HNM_min_prob + (1 - self.HNM_min_prob) / 2
        
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
        
        print "[INFO] Non-maxima suppression... "
        
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
    
    # TODO:
    def cross_validation(self, cv_scorings=None,
                         extract_new_features=False):
        """ Provede cross validaci na trenovacich datech """
        
        self.dataset.log_info("[INFO] Cross validation...")
        
        # pokud nejsou definovane scorings, tak je nacist z configu
        if cv_scorings is None:
            cv_scorings = self.config["cv_scorings"]
        
        if extract_new_features:
            
            self.extractor.features = {}
            self.dataset.create_dataset_CT()
            # rovnou volame vects, abychom mohli nastavit transform mode
            self.extractor.extract_feature_vects(multiple_rois=False, 
                                                 save_features=False,
                                                 mode="normal")
            self.create_training_data(features=self.extractor.features)
                                             
        else:
            self.create_training_data()
            
        X, y = self.data, self.labels
        
        print "Celkem dat: "
        print "   " + str( len([s for s in y if s > 0]) ) + " pozitivnich"
        print "   " + str( len([s for s in y if s < 0]) ) + " negativnich"
        
        # vypocet skore -> hodnoty test_ odpovidaji hodnotam cross_val_score
        scores = cross_validate(self.get_new_classifier(),  # vytvori novy klasifikator
                                X, y,
                                scoring=cv_scorings)
        for key in scores.keys():
            scores[key] = list(scores[key])
        print "[RESULT] Vysledne skore: ", scores
        
        # zapsani vysledku
        self.scores.append(scores)
        # ulozeni vysledku ohodnoceni
        self.dataset.zapis_json(scores, self.config["evaluation_path"]+"train_evaluation.json")
        # ulozeni do seznamu typu provedenych ohodnoceni
        self.evaluation_modes.append("train")
        # zalogovani zpravy o ukonceni
        self.dataset.log_info("      ... Hotovo.")
    
    # TODO:
    def evaluate_test(self, to_train=False):
        """ Ohodnoti vykon klasifikatoru na testovacich datech """
        
        self.dataset.log_info("[INFO] Classifier performance evaluation...")
        
        self.extractor.features = {}
        
        self.dataset.orig_images, self.dataset.negatives = [], []
        positives, negatives = self.get_test_data()
        
#        print positives
#        print negatives
        print len(positives), "pozitivnich a ", len(negatives), " negativnich",
        print "obrazku"
        
        self.extractor.n_negatives = len(negatives) + 1
        self.extractor.n_negative_patches = 4
        # rovnou volame vects, abychom mohli nastavit transform mode
        self.extractor.extract_feature_vects(multiple_rois=False, 
                                             save_features=False,
                                             mode="transform")
                                        
        
        self.create_training_data(mode="test", features=self.extractor.features)
        X, y = self.data, self.labels
        # pripadne trenovani
        if to_train:
            self.train()
        # pokud nechceme trenovat, ale zaroven nemame natrenovany klasifikator,
        # tak nacteme jiz natrenovany klasifikator ze souboru
        elif self.test_classifier is None:
            self.test_classifier = self.load_classifier()
            
        print "Celkem dat: "
        print "   " + str( len([s for s in y if s > 0]) ) + " pozitivnich"
        print "   " + str( len([s for s in y if s < 0]) ) + " negativnich"
        
        # ohodnoceni
        scores = dict()
        # vytvoreni porovnavacich dat
        y_pred = self.test_classifier.predict(X)
        # pocitani skore
        for metric_method in self.evaluation_test_metrics:
            scores[metric_method.__name__] = [metric_method(y_pred, y)]
            print "[RESULT] Vysledne skore podle "+metric_method.__name__+": ",
            print scores[metric_method.__name__]
        
        print "[RESULT] Confussion matrix: " 
        print confusion_matrix(y, y_pred)
        print "_____________________________"
        
        # zapsani vysledku
        self.scores.append(scores)
        # ulozeni vysledku ohodnoceni
        self.dataset.zapis_json(scores, self.config["evaluation_path"]+"test_evaluation.json")
        # ulozeni do seznamu typu provedenych ohodnoceni
        self.evaluation_modes.append("test")
        # znovuvytvoreni datasetu
        self.dataset.create_dataset_CT()
        # obnoveni extractoru
        self.extractor.count_number_of_negatives()
        # zalogovani zpravy o ukonceni
        self.dataset.log_info("      ... Hotovo.")
    
    
    def evaluate(self, mode='test', cv_scorings=['accuracy'], to_train=False):
        """ Bud provede cross validaci dat a nebo ohodnoti natrenovany 
        klasifikator na testovacich datech 
        
        Arguments:
            mode -- test  -> provede ohodnoceni modleu na testovacich datech
                 -- train -> provede cross validaci na trenovacich datech
            cv_scorings -- kriteria hodnotici funkce
            to_train -- zda se ma natrenovat klasifikator
        """
        
        # ohodnoti jiz natrenovany klasifikator na testovacich datech
        if mode == "test":
            self.evaluate_test(to_train=to_train)
            
        # provede cross-validaci
        elif mode == "train":
            self.cross_validation(cv_scorings=cv_scorings)
    
    
    def covered_by_artefact(self, mask_frame):
        """ Vrati indikator, zda je box vyplnen artefaktem ci nikoliv """
        
        # vypocet pokryti boxu a jeho stredu artefaktem
        bb_artefact_coverage = fe.artefact_coverage(mask_frame)
        bb_artefact_center_coverage, _ = fe.artefact_center_ellipse_coverage(mask_frame)
        # nastaveni prahu
        # TODO: cist z configu
        min_ac = 0.2    # minimalni pokryti boxu artefaktem
        min_acc = 0.6   # minimalni pokryti stredu boxu artefaktem
        # vrati logicky soucin techto dvou podminek
        return bb_artefact_coverage >= min_ac and bb_artefact_center_coverage >= min_acc
        
    
    def evaluate_nms_results_overlap(self):
        """ Ohodnoti prekryti vyslednych bounding boxu s artefakty """
        
        # pokud jese zadne vysledky nemame, tak nacteme existujici
        if len(self.test_results_nms.keys()) == 0:
            self.test_results_nms = self.dataset.precti_json(self.config["result_path"]+"results_nms.json")
        
        # inicializace statistik
        TP, TN, FP, FN = 0, 0, 0, 0

        for imgname, boxes in self.test_results_nms.items():
            
            # vypocet statistik pro dany obrazek
            TP0, TN0, FP0, FN0 = 0, 0, 0, 0
            
            # nacteni obrazku a masky
            img = self.dataset.load_image(imgname)
            mask = fm.get_mask(imgname, self.config)
            # oriznuti obrazku a masky -> takhle se to dela u augmentovanych
            img, mask = fe.cut_image(img, mask)
            
            # olabelovani artefaktu
            imlabel = fe.label(mask)
            # obarveni mist bez artefaktu na 0
            imlabel[(mask==0) | (mask==2)] = 0
            # vytvoreni prazdneho obrazku
            blank = np.zeros(img.shape)
            # ziskani indexu artefaktu
            artefact_ids = np.unique(imlabel)[1:]
            # seznam boxu, ktere pokryvaji nejaky artefakt
            covered_box_ids = list()
            
            # prochazeni vsech artefaktu
            for i in artefact_ids:
                
                covered_by_bb = False
                
                for j, (y, h, x, w) in enumerate(boxes):
                    # obarveni oblasti boxu
                    blank[y:h, x:w] = 1
                    # vypocet pixelu artefaktu celeho a v boxu
                    na = np.sum((imlabel==i).astype(int))
                    nab = np.sum((imlabel==i) & (blank==1))
                    # vypocet zastoupeni bb v artefaktu
                    artefact_bb_coverage = float(nab)/na
                    
                    # pokud je artefakt alespon z poloviny pokryt boxem
                    if artefact_bb_coverage >= 0.5:
                        
                        covered_box_ids.append(j)
                        # vytazeni frmau masky
                        mask_frame = mask[y:h, x:w]
                        # pokud jsou pokryty artefaktem -> TP, jinak FP
                        if self.covered_by_artefact(mask_frame):
                            TP += 1
                            TP0 += 1
                            covered_by_bb=True
                        else:
                            FP += 1
                            FP0 += 1
                    # znovu prebarveni pomocneho framu zpatky na 0      
                    blank[y:h, x:w] = 0
                
                # pokud neni pokryt zadnym boxem alespon z poloviny
                if not covered_by_bb:
                    FN += 1
                    FN0 += 1
            
            # prochazeni zatim neprohlendutych boxu
            for j in range(len(boxes)):
                if not j in covered_box_ids:
                    # vytazeni boxu
                    y, h, x, w = boxes[j]
                    mask_frame = mask[y:h, x:w]
                    # pokud jsou pokryty artefaktem -> TP, jinak FP
                    if self.covered_by_artefact(mask_frame):
                        TP += 1
                        TP0 += 1
                    else:
                        FP += 1
                        FP0 += 1
                        
            print TP0, TN0, FP0, FN0
            
        # finalni vyhodnoceni
        recall = float(TP) / (TP + FN)
        precision = float(TP) / (TP + FP)
        FPC = float(FP) / len(self.test_results_nms.keys())
        
        print "[RESULT] Celkove vysledky pro "+str(len(self.test_results_nms.keys()))+" obrazku:"
        print "         TP:", TP
        print "         TN:", TN
        print "         FP:", FP
        print "         FN:", FN
        print "        TPR:", recall
        print "  precision:", precision
        print "        FPC:", FPC
        
        results_to_save = {"TP": TP, "TN": TN, "FP": FP, "FN": FN,
                           "TPR": recall, "recall": recall,
                           "precision": precision, "FPC": FPC}
        
        self.dataset.zapis_json(results_to_save, 
                                self.config["evaluation_path"]+"nms_overlap_evaluation.json")
        
        return TN, FP, FN, TP       

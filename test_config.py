# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:51:56 2017

@author: Mirab
"""

import feature_extractor as fe
import data_reader as dr
import file_manager as fm

import re
import os
import cv2
import copy
import time
import datetime as dt

import skimage
from skimage.feature import hog as hogg
from skimage import exposure, data
#from skimage.filters import roberts, sobel, scharr, prewitt
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, recall_score, precision_score
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import PCA as PCA
from sklearn.decomposition import TruncatedSVD as DEC
from sklearn.feature_selection import VarianceThreshold as VT
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif


class Tester():
    
    def __init__(self, configpath="configuration/", configname="CT.json"):
        
        # cesta k datum
        self.pos_path = "datasets/PNG/datasets/frames/positives/"
        self.neg_path = "datasets/PNG/datasets/frames/negatives/"
        self.hnm_path = "datasets/PNG/datasets/frames/HNM/"
        
        # Inicializace
        self.manager = fm.Manager()
        self.extractor = fe.HOG()
        self.dataset = self.extractor.dataset
        #self.dataset.create_dataset_CT()
        self.config = self.dataset.config
        
        self.parentname = "extractor_test_results/All/"
        self.childname = ""
        self.childname_hog = ""
        
        self.blacklist = list()
        
        
    def create_paths(self, parentpath="extractor_test_results/All/"):
        """ Vytvori vsechny potrebne slozky a nektere soubory do nich ulozi """
    
        parentpath = self.parentname    
        
        paths_to_create = ["evaluation", "data_generator", "scripts",
                           "scripts/boxes", "scripts/config", "vars"]
                           
        for path in paths_to_create:
            self.manager.make_folder(parentpath + path)
        
        # zkopirovani skriptu
        fm.copyfile("test_config.py", parentpath+"test_config.py")
        # konfigurace a bounding_boxy
        fm.copyfile(self.config["annotations_path"],
                    parentpath+"scripts/boxes/boxes.json")
        fm.copyfile(self.dataset.config_path,
                    parentpath+"scripts/config/CT.json")
        # ulozeni vsech skriptu
        scripts = [name for name in fm.os.listdir(".") if name.endswith('.py')]
        for script in scripts:
            fm.copyfile(script, parentpath+"scripts/"+script)
    
    
    def backup_test_results(self, positives, negatives, 
                            targetname="extractor_test_results/"):
        """ Zalohuje vsechny vysledky testu """
        
        print "[INFO] Zalohuji vysledky...", 
        
        t = time.time()
        tstamp = str(dt.datetime.fromtimestamp(t))
        tstamp = re.sub(r'\s', '__', tstamp)
        tstamp = re.sub(r'[\:\.]', '-', tstamp)
    
        destination = targetname + tstamp + "/"
        
        # zapis nazvu obrazku
        fnames = {"positives": positives,
                  "negatives": negatives}
        dr.zapis_json(fnames, self.parentname+"image_names.json")
        # zapsani preskocenych konfiguraci
        dr.zapis_json({"skipped_configs": self.blacklist}, self.parentname+"blacklist.json")
        
        # zkopirovani cele cesty
        fm.copytree(targetname+"All", destination)
            
        print "Hotovo"
    
    
    def get_new_classifier(self):
        """ Vytvori a vrati instanci SVC """
        
        return SVC(kernel="linear", C = 0.15, probability=True, random_state=42)
    
    
    def get_methodname_old(self, method):
        """ Vrati stringovy label, ktery popisuje metodu """
        
        name = str(method)
        name = re.sub("[\(\)\<\>]", "#", name)
        name = re.sub("[\,\s,\:,\.\'\"]", "-", name)
        
        return name
    
    
    def get_methodname(self, method):
        """ Vrati popisek metody """
        
        name = str(method)
        method_names = ["PCA", "SelectKBest"]
        num_keywords = ["n_components=", "k="]
        func_keywords = ["function\s+", "random_state="]
        n = ""
    
        for keyword in num_keywords:
            num_label = re.findall(keyword+"\d+\.*\d*", name)
            if len(num_label) >= 1:
                n = re.sub("\.+", "-", num_label[0])
                break
    
        for keyword in func_keywords:
            key_label = re.findall(keyword+"\S+", name)
            if len(key_label) >= 1:
                new = re.sub("[\.\_]", "-", key_label[0])
                new = re.sub("\s", "=", new)
                new = re.sub("\,", "", new)
                n = n + "__" + new
        
        for method_name in method_names:
            if method_name in str(method):
                return method_name + "__" + n
        
        return "unknown_method"
    
    
    # TODO: implementovat
    def select_n_best(decompositions, i, X_shape):
        """ Zjisti, kolik features nechala jedna metoda (PCA)
        a jedne SelectKbest nastavi n_components na stejne cislo """
        pass
    
    
    def show_pca_vars(self, X, y, fvlp):
        """ Spocita a seradi vlastni cisla kovariancni matice 
        a vykresli je  spolu s jejich kumulativni summou a mezemi.
        Vse pote ulozi jako graf png. """
        
        # natrenovani PCA
        pca = PCA()
        pca.fit(X, y)
        
        # vykresleni vlastnich cisel
        plt.ylim(0, 1)
        plt.plot(pca.explained_variance_ratio_.cumsum(), "b", lw=2)
        plt.plot(pca.explained_variance_ratio_ / pca.explained_variance_ratio_ .max(), "r", lw=2)
        plt.grid()
        plt.title(self.childname_hog + "_fvlp=" + str(fvlp))
        
        # barvy pro primky prahu
        col = ["b", "r", "g", "c", "m", "y"]
        thrs = [0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
        
        # vykreslovani primek prahu
        for i, thr in enumerate(thrs):
            n_components = len(np.where(pca.explained_variance_ratio_.cumsum() <= thr)[0])
            plt.axvline(n_components,
                        linestyle='-.', 
                        label="",
                        color=col[i],
                        lw=2)
        # ulozeni obrazku                
        plt.savefig(self.parentname+"/vars/"+self.childname_hog+".png")
        #plt.show()
        plt.close()
        
    # TODO: dat tam cv = kfold, kde nsplits=3
        # pro mala data 2x a prumer
        # cross_val predict a confussion matrix
    def cross_validation(self, X, y, cv_scorings=None, random_state=42):
        """ Provede cross-validaci pro dana data """
        
        #self.dataset.log_info("[INFO] Cross validation...")
        print "[INFO] Cross validation..."
        
        # pokud nejsou definovane scorings, tak je nacist z configu
        if cv_scorings is None:
            cv_scorings = self.config["cv_scorings"]
        
        print "  Celkem dat: "
        print "     " + str( len([s for s in y if s > 0]) ) + " pozitivnich"
        print "     " + str( len([s for s in y if s < 0]) ) + " negativnich"
        print "     Celkovy tvar dat: ", X.shape
        
        # rozdeleni dat na 3 splity -> shuffle nutne !!!
        kfold = KFold(n_splits=3, shuffle=True, random_state=random_state)
        # vypocet predikovanych hodnot y
        y_pred = cross_val_predict(self.get_new_classifier(),  # vytvori novy klasifikator
                                   X, y,
                                   cv=kfold,
                                   n_jobs=-1)
        # vypocet confusion matrix
        TN, FP, FN, TP = confusion_matrix(y, y_pred).astype(int).ravel()
        # vypocet skore podle ruznych metrik
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        accuracy = accuracy_score(y, y_pred)
        
        # vypsani vysledku
        print "  [RESULT] Vysledne skore: "
        print "  precision: ", precision
        print "     recall: ", recall
        print "         f1: ", f1
        print "   accuracy: ", accuracy
        print "         TN: ", TN
        print "         FP: ", FP
        print "         FN: ", FN
        print "         TP: ", TP
        # ulozeni vysledku
        scores= {"precision": precision,
                 "recall": recall,
                 "f1": f1,
                 "accuracy": accuracy,
                 "TP": TP, "FP": FP, "TN": TN, "FN": FN}
                 
        # davani hodnot do listu                   
        for key in scores.keys():
            scores[key] = list([scores[key]])
                    
        # ulozeni vysledku ohodnoceni
        dr.zapis_json(scores, self.parentname+"/evaluation/CV_"+self.childname+".json")
        
        # zalogovani zpravy o ukonceni
        #self.dataset.log_info("      ... Hotovo.")
        
        print "Hotovo."
        
        
    
    def fit_methods(self, positives, negatives, decompositions, n_for_fit=2000,
                    fvlp=0, to_dec=False, raw=False):
        """ Extrahuje data pro redukc dimenzionality """
        
        print "[INFO] Extrahuji data pro redukci dimenzionality... "
        print "       PCA mode = ", self.extractor.PCA_mode
        
        # pro opravdu velke vektory snizit pocet dat
        if fvlp > 2000:
            n_for_fit = 1500
        if fvlp > 2500:
            n_for_fit = 1000
        if fvlp > 4000:
            n_for_fit = 700
        if fvlp > 5000:
            n_for_fit = 500
        
        P = len(positives)
        N = len(negatives)    
        
        each_img = max( (P + N) // (n_for_fit * 2), 1)
        # pokud jde o surova data, tak beru zady obrazek
        if to_dec and raw:
            each_img = 1
        
        Xr = list()
        yr = list()
        
        # nacitani dat -> v hogu je PCA_mode na True
        # vybiram poze nazde each_img - te obrazky
        for i, imgname in enumerate(positives):
            if i % each_img == 0:
                img = dr.load_image(imgname)
                feature_vect = self.extractor.extract_single_feature_vect(img)
                Xr.append(feature_vect)
                yr.append(1)
        
        for i, imgname in enumerate(negatives):
            if i % each_img == 0:
                img = dr.load_image(imgname)
                feature_vect = self.extractor.extract_single_feature_vect(img)
                Xr.append(feature_vect)
                yr.append(-1)
        
        Xr = np.vstack(Xr)
        yr = np.array(yr)
        
        # natrenovani vsech metod dekompozice nebo fetaure selekce
        for d, dec in enumerate(decompositions):
            dec.fit(Xr, yr)
            # pokud jde o PCA a nechceme pouze vykreslovat variance
            if "PCA" in str(dec) and not to_dec:
                X_new = dec.transform(Xr[0:2])
                # nastaveni poctu komponent jedne metody SelectKBest 
                # podle tvaru transformovanych dat
                decompositions[d+len(decompositions)//2] = SelectKBest(f_classif, k=X_new.shape[1])
        
        # pripadne vykresleni vlastnich cisel metod
        if to_dec: self.show_pca_vars(Xr, yr, fvlp)
         
        print "  Raw data shape: ", Xr.shape
        print "  Celkem dat: "
        print "     " + str( len([s for s in yr if s > 0]) ) + " pozitivnich"
        print "     " + str( len([s for s in yr if s < 0]) ) + " negativnich"
        print "Hotovo"
        
        return decompositions
    
    
    def extract_data(self, positives, negatives):
        """ Ze seznamu obraku (framu) extrahuje HoGy """
        
        print "[INFO] Extrahuji data..."
        print "       PCA_mode = ", self.extractor.PCA_mode
    
        X = list()
        y = list()
                    
        for i, imgname in enumerate(positives):
            img = dr.load_image(imgname)
            feature_vect = self.extractor.extract_single_feature_vect(img)
            X.append(feature_vect)
            y.append(1)
    
        for i, imgname in enumerate(negatives):
            img = dr.load_image(imgname)
            feature_vect = self.extractor.extract_single_feature_vect(img)
            X.append(feature_vect)
            y.append(-1)
        
        X = np.vstack(X)
        y = np.array(y)
        
        print "  Data shape: ", X.shape
        print "  Celkem dat: "
        print "     " + str( len([s for s in y if s > 0]) ) + " pozitivnich"
        print "     " + str( len([s for s in y if s < 0]) ) + " negativnich"
        print "Hotovo"
            
        return X, y


if __name__ =='__main__':
    
    to_cv = bool(1)
    to_dec = bool(0)
    
    t = time.time()
    
    # inicializace
    tester = Tester()
    hog = tester.extractor
    
    # vytvoreni potrebnych cest
    tester.create_paths()
    
    # nacteni seznamu obrazku
    positives = [tester.pos_path + imgname for imgname in os.listdir(tester.pos_path) if imgname.endswith('.png')]#  and not ('AFFINE' in imgname)]
    negatives = [tester.neg_path + imgname for imgname in os.listdir(tester.neg_path) if imgname.endswith('.png')]#  and not ('AFFINE' in imgname)]        
    #hnms = [tester.hnm_path + imgname for imgname in os.listdir(tester.hnm_path) if imgname.endswith('.png')]# and not ('AFFINE' in imgname)]
    
    """ Nastaveni parametru """
    # HoG
    oris = [12, 16, 20]
    ppcs = [10, 8, 6, 4]
    cpbs = [2, 3, 4]

    # doporucene
    oris = [6, 9, 12]#, 15]
    ppcs = [10, 8, 6, 4]
    cpbs = [1, 2, 3]
    
    # vyfiltrovane
#    oris = [6, 9, 12]
#    ppcs = [8, 6, 4]
#    cpbs = [1, 2, 3]
#    
#    oris = [9, 12]
#    ppcs = [6, 4]
#    cpbs = [2]

    # nejvetsi data nejdrive
    oris, ppcs, cpbs = oris[::-1], ppcs[::-1], cpbs[::-1]
    
#    oris = [12, 16]
#    ppcs = [10, 8, 6]
#    cpbs = [2, 3]
    
#    oris = [6]
#    ppcs = [10]
#    cpbs = [1]
    
    # musi byt presne napasovane na seznam decompositions !!!
    dec_fvls = [10, 32, 128, 512]# nastavt na nulu, pokud nenastavujeme pocet features
    # redukce vektoru priznaku
#    decompositions = [PCA(n_components=10), 
#                      PCA(n_components=32),
#                      PCA(n_components=128),
#                      PCA(n_components=64)][:1]
    
    # vzdy musi byt N PCA a za nimi nasledovat stejny pocet SelectKbest
    decompositions = [PCA(0.8), 
                      PCA(0.9),
                      SelectKBest(),
                      SelectKBest()]
    
    # dodelal jsem 20-6-3
    """ Proces testovani vsech parametru """
    max_iters = len(oris) * len(ppcs) * len(cpbs) * len(decompositions)
    iters = 0
    
    for ori in oris:
        for ppc in ppcs:
            for cpb in cpbs:
                
                print "Testuje se: ", ori, "-", ppc, "-", cpb
                
                # nastaveni kongigurace feature extractoru (HoGu)
                hog.orientations = ori
                hog.pixels_per_cell = (ppc, ppc)
                hog.cells_per_block = (cpb, cpb)
                
                # nastaveni nazvu slozky
                tester.childname_hog = "ori="+str(ori)+"_ppc="+str(ppc)+"_cpb="+str(cpb)
                
                # spocteni velikocti fv
                fvlp = ori * cpb**2 * ( (hog.sliding_window_size[0] // ppc) - (cpb - 1) )**2
                print "Predpokladana velikost feature vektoru: ", fvlp
                # pokud bude fv moc dlouhy, tak fitnout jen na casti a pak transformovat kazdy
                partially = fvlp >= 1300
                # netestovat extremne rozmerne feature vektory 
                # -> ulozit do blacklistu, ze jsem je netestoval
                if fvlp >= 10000:
                    black = {"ori": ori,
                             "ppc": ppc,
                             "cpb": cpb,
                             "fvlp": fvlp}
                    tester.blacklist.append(black)
                    # pricteni prozkoumanyhc konfiguraci
                    iters += len(decompositions)
                    continue
                # pokud budou male vektory, tak muzeme extrahovat originalni
                # data a tim padem nechceme PCA provadet u extrakce
                hog.PCA_mode = partially
                
                # pokud zkoumam jen pca, tak fitnout pca
                if to_dec:
                    hog.PCA_mode = False
                    decompositions = tester.fit_methods(positives, 
                                                        negatives, 
                                                        [PCA(n_components=2)],
                                                        fvlp=fvlp,
                                                        to_dec=to_dec,
                                                        raw = not partially)
                    hog.PCA_mode = True
                    continue

                if partially:
                    # zatim nastavim PCA_mode v extractoru na False -> 
                    #        -> to fitnuti si totiz udelam sam
                    hog.PCA_mode = False
                    # natrenovani metody
                    decompositions = tester.fit_methods(positives, 
                                                        negatives, 
                                                        decompositions,
                                                        fvlp=fvlp,
                                                        to_dec=to_dec)
                    # ted uz nastavim PCA mode na True, aby mi to vyhazovalo
                    #     transformovane vektory podle dane metody
                    hog.PCA_mode = True
                    # testovani                        
                    for decomposition in decompositions:                       
                        # zjisteni velikosti redukovaneho vektoru
                        dec_name = tester.get_methodname(decomposition)
                        # nastaveni metody redukce dimenze
                        hog.PCA_object = decomposition
                        # doplneni nazvu slozky
                        tester.childname = tester.childname_hog + "_" + dec_name
                        # extrakce jiz transformovanych dat
                        X, y = tester.extract_data(positives, negatives)
                        # cross validace
                        if to_cv: tester.cross_validation(X, y)
                        
                        # vypsani informace o progresu
                        iters += 1
                        act_time = time.time() - t
                        print "Time: ",  act_time,
                        print " - ", iters, " z ", max_iters
                        print "Predpokladana doba trvani: ", 
                        print "%.2f" % ((float(max_iters) / iters) * act_time / 3600),
                        print " hodin."
                        
                else:
                    # extrakce originalnich feature vektoru
                    X_raw, y = tester.extract_data(positives, negatives)
                    print "[INFO] Data shape: ", X_raw.shape
                    # testovani 
                    for d, decomposition in enumerate(decompositions):
                        # zjisteni velikosti redukovaneho vektoru
                        dec_name = tester.get_methodname(decomposition)
                        # doplneni nazvu slozky
                        tester.childname = tester.childname_hog + "_" + dec_name
                        # redukce dimenzionality na celych datech
                        X = decomposition.fit_transform(X_raw, y)
                        # nastaveni poctu komponent odpovidajici metody SelectKBest
                        if "PCA" in dec_name:
                            # nastaveni poctu komponent jedne metody SelectKBest 
                            # podle tvaru transformovanych dat
                            decompositions[d+len(decompositions)//2] = SelectKBest(f_classif, k=X.shape[1])
                            
                        # cross validace
                        if to_cv: tester.cross_validation(X, y)
                        
                        # vypsani informace o progresu
                        iters += 1
                        act_time = time.time() - t
                        print "Time: ",  act_time,
                        print " - ", iters, " z ", max_iters
                        print "Predpokladana doba trvani: ", 
                        print "%.2f" % ((float(max_iters) / iters) * act_time / 3600),
                        print " hodin." 
    
    #dr.save_obj((X_raw, y), tester.parentname+"data.pklz")
    
    # zaloha vysledku
    tester.backup_test_results(positives, negatives)
    
    print "Celkovy cas: ", time.time() - t     

           
    """ obsolete """ 

#    # prebarvovani
#    colorings = [None, 25, 29, 33]
#    
#    # preprocessing
#    processing_methods = [(cv2.bilateralFilter, [[9, 35, 35]]),
#                          (cv2.medianBlur, [7, 9, 11, 13])]

#    for coloring in colorings:
#        # rozmyslet si, zda to nebudu testovat rucne
#        hog.background_coloring_ksize = coloring
#        if coloring is None:
#            hog.to_color = False
#            

              
     # pro vsechny mozne metody processingu
#    for method, params in processing_methods: 
#        for param in params:
#            if "bilateral" in method.__name__:
#                processing_method = method(roi, 
#                                           param[0],
#                                           param[1],
#                                           param[2])
#            elif "median" in method.__name__:
#                processing_method = method(roi, param)

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 05 17:42:48 2018

@author: Mirab
"""

import csv
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import glob
import sys



import file_manager_metacentrum as fm


def open_csv(fname):
    data = list()
    with open(fname, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar=';')
        for row in reader:
            data.append(row)
    return data


def open_csv_tonda(fname):
    data = list()
    with open(fname, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar=' ')
        for row in reader:
            data.append(row)
    return data


def print_csv(data):
    for row in data:
        print(row)
        

def make_npdata(data, rows=(2,4)):
    npdata = list()
    for r, row in enumerate(data):
        nprow = list()
        for column in row[rows[0]:rows[1]]:
            nprow.append(float(re.sub("\,", ".", column)))
        npdata.append(nprow)
    return np.vstack(npdata)
    
    
def plot_CSVlog(fname):
    fname = "classification/Keras/experiments/no_aug_structured_data-liver_only/callbacks_6epochs/logs/csv_logger_fit.csv"
    
    data = open_csv(fname)
    categories = tuple(enumerate(data[0]))
    npdata = make_npdata(data[1:], rows=[0,5])
    
    draw_CSVlog(npdata, categories)


def draw_CSVlog(data, categories):
    """ Vykresli vyvoj accuracy po danych epochach """
    
    plt.figure(figsize=(10,12))
    plt.title("Accuracy trend")
    plt.plot(data[:, 0], data[:, 1], label=categories[1][1], color="b", lw=2)
    plt.plot(data[:, 0], data[:, 3], label=categories[3][1], color="r", lw=2)
    plt.grid()
    plt.legend(bbox_to_anchor=(1.0, 0.25), loc=1, borderaxespad=0.)
    plt.show()


def read_txt(fname):
    lines = None
    with open(fname, "r") as file:
        lines = file.readlines()
        file.close()
    return lines
    

def read_batchesdata(foldername):
    """ Nacte log batchu a vrati jej jako list slovniku """
    
    fname = foldername
    
    if not foldername.endswith(".log"):
        fname = foldername + "batches.log"
    
    data = list()
    lines = read_txt(fname)
        
    for line in lines:
        data.append(fm.json.loads(line))
        
    return data


def draw_batches_parallel(foldername="classification/Keras/experiments/no_aug_structured_data-liver_only/callbacks_6epochs/logs/"):
    """ Vykresli vyvoj train accuracy a loss za sebe po batchich """
    
    batchesdata = read_batchesdata(foldername)
    
    fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
    ax1.set_title("Train accuracy per batches")
    ax2.set_title("Train loss per batches")

    ax1.set_ylim((0.6, 1.1))
    ax2.set_ylim((0, 1.1))
    
    batches_acc = [event["acc"] for event in batchesdata if "batch" in event.keys()]
    batches_loss = [event["loss"] for event in batchesdata if "batch" in event.keys()]

    n = 0
    n_old = 0
    xl = 0
    i = 0
    colors = ["y", "c", "g", "m", "r", "b", "k"]
    for event in batchesdata:
        if "epoch" in event.keys():
            ax1.plot(batches_acc[n_old:n], lw=1, color=colors[i])
            ax2.plot(batches_loss[n_old:n], lw=1, color=colors[i])
            i += 1
            xl = max(xl, n-n_old)
            n_old = n
            continue
        n += 1
    
    i += 1
    ax1.plot(batches_acc[n_old: n], lw=1, color=colors[i], label="train_acc")
    ax2.plot(batches_loss[n_old: n], lw=1, color=colors[i], label="train_loss")
    
    ax1.set_xlim((-xl/10, 1.1*xl))
    ax2.set_xlim((-xl/10, 1.1*xl))
    
    ax1.grid()
    ax2.grid()
    #plt.legend(bbox_to_anchor=(1.0, 0.45), loc=1, borderaxespad=0.)
    plt.show()


def draw_batches_concat(foldername="classification/Keras/experiments/no_aug_structured_data-liver_only/callbacks_6epochs/logs/",
                        to_save=False, detailed=False):
    """ Vykresli vyvoj train accuracy a loss za sebe po batchich """
    
    batchesdata = read_batchesdata(foldername)
    
    plt.figure()
    plt.title("Train accuracy and loss per batches")
    plt.xlim((-len(batchesdata)/10, 1.1*len(batchesdata)))
    plt.ylim((0, 1.1))
    
    batches_acc = [event["acc"] for event in batchesdata if "batch" in event.keys()]
    batches_loss = [event["loss"] for event in batchesdata if "batch" in event.keys()]
    
    plt.plot(batches_acc, lw=1, color="b", label="train_acc")
    plt.plot(batches_loss, lw=1, color="r", label="train_loss")
    
    epochs_acc = []
    epochs_loss = []
    ns = []
    n = 0
    for event in batchesdata:
        if "epoch" in event.keys():
            ns.append(n)
            epochs_acc.append(event["acc"])
            epochs_loss.append( event["loss"])
            plt.axvline(x=n, color="g", linestyle="--", lw=2)#, label="epoch "+str(len(epochs_acc)))
        n += 1
    
    plt.grid()
    plt.legend(bbox_to_anchor=(1.0, 0.45), loc=1, borderaxespad=0.)
    
    if to_save:
        plt.savefig(foldername + "/batches_concat.png")
        
    if detailed:
        plt.figure()
        plt.title("Train accuracy")
        plt.plot(batches_acc, lw=1, color="b", label="train_acc")
        plt.xlim((-len(batchesdata)/10, 1.1*len(batchesdata)))
        plt.ylim((0.9, 1.05))
        plt.grid()
        plt.legend(bbox_to_anchor=(1.0, 0.35), loc=1, borderaxespad=0.)
        if to_save:
            plt.savefig(foldername + "/batches_concat_acc.png")
        
        plt.figure()
        plt.title("Train loss")
        plt.plot(batches_loss, lw=1, color="r", label="train_loss")
        plt.xlim((-len(batchesdata)/10, 1.1*len(batchesdata)))
        plt.ylim((0, 0.3))
        plt.grid()
        plt.legend(bbox_to_anchor=(1.0, 0.85), loc=1, borderaxespad=0.)
        if to_save:
            plt.savefig(foldername + "/batches_concat_loss.png")


def draw_batches_concat_all(foldername="experiments/aug_structured_data-liver_only/"):
    """ Pro vsechny experimenty ve slozce provede vykresleni logu """
    
    if not foldername.endswith("/"):
        foldername = foldername + "/"
        
    path_to_experiments = foldername
    folders = glob.glob(path_to_experiments+"*")
    
    print("[INFO] Vykresluji prubehy accuracy a loss...")
    
    for f in folders:
        print("    - " + str(f))
        fname = f + "/logs/"
        draw_batches_concat(foldername=fname, to_save=True, detailed=True)
        
    print("[INFO] Hotovo.")
    
        
if __name__ == "__main__":
    
    """
    fname = "classification/Keras/experiments/no_aug_structured_data-liver_only/callbacks_6epochs/logs/csv_logger_fit.csv"
    
    data = open_csv(fname)
    categories = tuple(enumerate(data[0]))
    npdata = make_npdata(data[1:], rows=[0,5])
    
    print categories
    """
    
    """
    # CSV log
    #draw_CSVlog(npdata, categories)
    
    # batches
    #draw_batches_concat()
    #draw_batches_parallel()
    """
    
    matplotlib.use('Agg')
    
    foldername = "experiments/aug_structured_data-liver_only/"
    
    if sys.version.startswith("3"):
        matplotlib.use('Agg')
        
    elif sys.version.startswith("2"):
        foldername = "classification/Keras/experiments/aug_structured_data-liver_only/"
        print("[INFO] Python 2")
    
    draw_batches_concat_all(foldername=foldername)
    
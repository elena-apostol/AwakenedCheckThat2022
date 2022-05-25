# coding: utf-8

__author__      = "Elena-Simona Apostol; Ciprian-Octavian Truică"
__copyright__   = "Copyright 2022, Uppsala University"
__license__     = "GNU GPL"
__version__     = "0.1"
__email__       = "elena-simona.aportol@it.uu.se; ciprian-octavian.truica@it.uu.se"
__status__      = "Production"

# helpers
import time

# classification
import numpy as np
import pandas as pd
import sys
import os
import random as rnd
import math
from scipy import io as sio



from sentence_transformers import SentenceTransformer


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0" 


def getBARTEncoding(X):
    model = SentenceTransformer('facebook/bart-large')
    X_BART = model.encode(X)
    return X_BART

def getXLMEncoding(X):
    model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')
    X_XMLRoBERTa = model.encode(X)
    return X_XMLRoBERTa



if __name__ =="__main__":
    FIN_TRAIN = sys.argv[1]
    FIN_DEV = sys.argv[2]
    FIN_TEST = sys.argv[3]
    DIR_NAME = sys.argv[4] # the directory
    LANG = sys.argv[5] # en - English, de - German
    print(USE_CUDA)

    ds_train = pd.read_csv(FIN_TRAIN, sep=',', encoding = "utf-8")
    ds_dev =  pd.read_csv(FIN_DEV, sep=',', encoding = "utf-8")
    ds_test = pd.read_csv(FIN_TEST, sep=',', encoding = "utf-8")
    dataSet = pd.concat([ds_train, ds_dev], ignore_index=True, sort=False)
    print("Train shape", ds_train.shape)
    print("Dev shape", ds_dev.shape)
    print("Test shape", ds_test.shape)
    print("Train full shape", dataSet.shape)
    dataSet['our rating'] = dataSet['our rating'].str.lower()
    labels = dataSet['our rating'].unique()
    id2class = {'false': 0, 'partially false': 1, 'other': 2, 'true': 3}
    for label in labels:
        dataSet.loc[dataSet['our rating'] == label, 'label'] = id2class[label]


    print("No. classes", labels)
    print("id2class", id2class)

    dataSet["content"] = dataSet["text"] + " " + dataSet["title"]
    ds_test["content"] = ds_test["text"] + " " + ds_test["title"]

    X_train = dataSet['content'].astype(str).to_list()
    X_test = ds_test['content'].astype(str).to_list()
    X_test_id = ds_test['ID'].astype(str).to_list()
    y = dataSet['label'].astype(int).to_list()
    sio.savemat(os.path.join(DIR_NAME, 'labels.mat'), {'y': y})
    sio.savemat(os.path.join(DIR_NAME, 'D2V_ID_TEST.mat'), {'X': X_test_id})



    
    if LANG == 'en':
        start_time = time.time()
        X_BART_TRAIN = getBARTEncoding(X_train)
        X_BART_TEST = getBARTEncoding(X_test)
        print(len(X_BART_TRAIN))
        print(len(X_BART_TEST))
        end_time = time.time()
        print("Time taken extract BART Sentence Embeddings: ", end_time - start_time)

        sio.savemat(os.path.join(DIR_NAME, 'D2V_BART_TRAIN.mat'), {'X': X_BART_TRAIN})
        sio.savemat(os.path.join(DIR_NAME, 'D2V_BART_TEST.mat'), {'X': X_BART_TEST})
    
    elif LANG == 'de':
        start_time = time.time()
        X_XML_TRAIN = getXLMEncoding(X_train)
        X_XML_TEST = getXLMEncoding(X_test)
        print(len(X_XML_TRAIN))
        print(len(X_XML_TEST))
        end_time = time.time()
        print("Time taken extract XLM Sentence Embeddings: ", end_time - start_time)

        sio.savemat(os.path.join(DIR_NAME, 'D2V_XML_TRAIN.mat'), {'X': X_XML_TRAIN})
        sio.savemat(os.path.join(DIR_NAME, 'D2V_XML_TEST.mat'), {'X': X_XML_TEST})
    

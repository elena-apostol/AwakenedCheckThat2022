# coding: utf-8

__author__      = "Elena-Simona Apostol; Ciprian-Octavian Truică"
__copyright__   = "Copyright 2022, Uppsala University"
__license__     = "GNU GPL"
__version__     = "0.1"
__email__       = "elena-simona.aportol@it.uu.se; ciprian-octavian.truica@it.uu.se"
__status__      = "Production"


import os
import sys
import numpy as np
from scipy import io as sio

# helpers
import time

# import keras
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

epochs_n = 100
units = 100

if __name__ =="__main__":
    DIR_NAME = sys.argv[1] # the directory

    ##################### LABELS ############################
    id2class = {0: 'false', 1: 'partially false', 2: 'other', 3: 'true'}

    y_train = sio.loadmat(os.path.join(DIR_NAME, 'labels.mat'))['y'][0]
    
    x_train = sio.loadmat(os.path.join(DIR_NAME, 'D2V_XML_TRAIN.mat'))['X']
    x_test = sio.loadmat(os.path.join(DIR_NAME, 'D2V_XML_TEST.mat'))['X']
    ids = sio.loadmat(os.path.join(DIR_NAME, 'D2V_ID_TEST.mat'))['X']
    num_classes = len(np.unique(y_train))

    x_vec_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_vec_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    y_vec_train = to_categorical(y_train, num_classes=num_classes)

    #  BiLSTM
    start_time = time.time()
    input_layer = Input(shape=(x_vec_train.shape[1], x_vec_train.shape[2]), name = 'Input')
    lstm_layer = Bidirectional(LSTM(units = units), name='BiLSTM')(input_layer)
    output_layer = Dense(num_classes, activation='sigmoid', name='Output')(lstm_layer)

    model = Model(inputs=input_layer, outputs=output_layer, name="BiLSTM")

    if num_classes == 2:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())
    es = EarlyStopping(monitor='accuracy', mode='min', verbose=1, patience=20)

    history = model.fit(x = x_vec_train, y = y_vec_train, epochs=epochs_n, verbose=True, batch_size=64, callbacks=[es])

    y_pred = model.predict(x_vec_test, verbose=False)
    print(y_pred)
    y_p = np.argmax(y_pred, 1)
    print(y_p)
    end_time = time.time()

    exc_time = (end_time - start_time)
    print("BiLSTM Time", exc_time)

    labels = []
    for elem in y_p:
        labels.append(id2class[elem])

    with open("subtask3_german_awakened.tsv", "a") as f:
        f.write("public_id, predicted_rating\n")
        for idx in range(0, len(labels)):
            line = ids[idx].strip() + ", " + labels[idx] + "\n"
            f.write(line)
    

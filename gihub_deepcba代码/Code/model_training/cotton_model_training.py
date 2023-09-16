# -*- coding: utf-8 -*-
'''
@ author LiJ
@ version 1.0
@ date 2023 / 03 / 27 18: 46
@ Description: data encode for human
'''

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import Bio.SeqIO
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import backend as K

#take modelxxx.py from model_code
from model import buildModel

from data_encode import data_encode
from utils import show_train_history

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        tf.config.experimental
        # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.67)
        # config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        # session = tf.compat.v1.Session(config=config)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")
#tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy('float32')


def pearson(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)

def train_and_pre(file_name):
    save_name=file_name.split('_')[0]+'_'+file_name.split('_')[1]

    # -----------------load data----------------#
    np.random.seed(np.random.seed(1))
    dataset = pd.read_csv('../Data/%s' % (file_name), encoding='utf-8')
    gene1 = dataset['Annotation1'].values.astype('str')
    gene2 = dataset['Annotation2'].values.astype('str')
    seq1 = dataset['Seq1'].values.astype('str')
    seq2 = dataset['Seq2'].values.astype('str')
    exp=dataset['Expression'].values.astype('float')

    state = np.random.get_state()
    np.random.shuffle(gene1)
    np.random.set_state(state)
    np.random.shuffle(gene2)
    np.random.set_state(state)
    np.random.shuffle(seq1)
    np.random.set_state(state)
    np.random.shuffle(seq2)
    np.random.set_state(state)
    np.random.shuffle(exp)

    slide1 = int(len(gene1) * 0.6)
    slide2 = int(len(gene1) * 0.8)

    print(' data_encoding '.center(100, '-') + '\n')
    dna1 = data_encode(seq1)
    dna2 = data_encode(seq2)
    print(' encoding_success '.center(100, '$') + '\n')

    # ------------build model & train-----------#

    model = buildModel()

    model.compile(loss='mse',
                  optimizer=keras.optimizers.Adam(1e-3),
                  metrics=['accuracy', pearson])

    ckpt = keras.callbacks.ModelCheckpoint('../Model/%s.h5' %(save_name),
                                           monitor='val_pearson',
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=False,
                                           mode='max',
                                           period=1)

    history = model.fit([dna1[:slide1], dna2[:slide1]],
                        exp[:slide1],
                        epochs=500,
                        batch_size=64,
                        validation_data=([dna1[slide1:slide2], dna2[slide1:slide2]], exp[slide1:slide2]),
                        callbacks=[ckpt])

    show_train_history(history.history, s=save_name, locate='../Result/pic/')
    #    ROC(model, [dna1[slide1:], dna2[slide1:]], lab[slide1:],)

    # --------------------predict----------------------------#
    test_data = [dna1[slide2:], dna2[slide2:]]
    test_label = exp[slide2:]

    rslt = model.predict(test_data)
    array = np.array([gene1[slide2:],gene2[slide2:],test_label, rslt[:, 0]]).T
    df = pd.DataFrame(array, columns=['ann1','ann2','label', 'result'])
    df.to_csv('../Result/output_%s.csv' %(save_name), index=False)

files=os.listdir('../Data/')
for i in files:
    train_and_pre(i)
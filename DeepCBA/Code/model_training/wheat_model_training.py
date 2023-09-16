# -*- coding: utf-8 -*-
'''
@ author LiJ
@ version 1.0
@ date 2023 / 02 / 26 15: 11
@ Description: wheat exp data training
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

wheat_tss_geneID = []
wheat_tts_geneID = []
wheat_tss_seqs = []
wheat_tts_seqs = []

for x in Bio.SeqIO.parse('../../../../小麦专用数据/序列数据/wheat_HC_gene_tss_1.5k_seq.fasta', 'fasta'):
    wheat_tss_geneID.append(x.id.split('_')[0])
    wheat_tss_seqs.append(x.seq)

for x in Bio.SeqIO.parse('../../../../小麦专用数据/序列数据/wheat_HC_gene_tts_1.5k_seq.fasta', 'fasta'):
    wheat_tts_geneID.append(x.id.split('_')[0])
    wheat_tts_seqs.append(x.seq)


file_name = os.listdir('../Data/小麦/')
for i in tqdm(range(len(file_name))):
    # -----------------load data----------------#
    np.random.seed(1)
    wheat_data = pd.read_csv('../Data/小麦/' + file_name[i], encoding='utf-8')
    wheat_data_ann1 = wheat_data['Annotation1'].values.astype('str')
    wheat_data_ann2 = wheat_data['Annotation2'].values.astype('str')
    wheat_data_exp = wheat_data['exp'].values.astype('float')

    wheat_data_dnn1, wheat_data_dnn2 = data_encode(wheat_data_ann1, wheat_data_ann2, wheat_tss_geneID,wheat_tts_geneID,wheat_tss_seqs , wheat_tts_seqs)

    # if file_name[i][:2] == 'MH':
    #     wheat_data_dnn1, wheat_data_dnn2 = data_encode(wheat_data_ann1, wheat_data_ann2, MH_wheat_tss_geneID,
    #                                                  MH_wheat_tts_geneID, MH_wheat_tss_seqs, MH_wheat_tts_seqs)
    # else:
    #     wheat_data_dnn1, wheat_data_dnn2 = data_encode(wheat_data_ann1, wheat_data_ann2, ZS_wheat_tss_geneID,
    #                                                  ZS_wheat_tts_geneID, ZS_wheat_tss_seqs, ZS_wheat_tts_seqs)


    # ------------build model & train-----------#

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


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


    slide1 = int(len(wheat_data_ann1) * 0.6)
    slide2 = int(len(wheat_data_ann1) * 0.8)

    model = buildModel()
    # model.summary()
    model.compile(loss='mse',
                  optimizer=keras.optimizers.Adam(1e-3),
                  metrics=['accuracy', pearson])

    ckpt = keras.callbacks.ModelCheckpoint('../Model/'+file_name[i][:-4]+'.h5',
                                           monitor='val_pearson',
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=False,
                                           mode='max',
                                           period=1)

    history = model.fit(np.concatenate([wheat_data_dnn1[:slide1], wheat_data_dnn2[:slide1]],axis=2),
                        wheat_data_exp[:slide1],
                        epochs=500,
                        batch_size=64,
                        validation_data=(np.concatenate([wheat_data_dnn1[slide1:slide2], wheat_data_dnn2[slide1:slide2]],axis=2), wheat_data_exp[slide1:slide2]),
                        callbacks=[ckpt])

    show_train_history(history.history, s=file_name[i][:-4] + '_data_augment', locate='../Result/pic/')
    #show_train_history(history.history, s=zu_name, locate='../Result/pic/' + mode + '/')


    # --------------------predict----------------------------#
    test_data = np.concatenate([wheat_data_dnn1[slide2:], wheat_data_dnn2[slide2:]],axis=2)
    test_label = wheat_data_exp[slide2:]

    rslt = model.predict(test_data)
    array = np.array([wheat_data_ann1[slide2:], wheat_data_ann2[slide2:], test_label, rslt[:, 0]]).T
    df = pd.DataFrame(array, columns=['Annotation1', 'Annotation2', 'exp', 'pre'])
    df.to_csv('../Result/pred_'+ file_name[i], index=False)

# # -----------------load data----------------#
#
# np.random.seed(1)
# dna1 = np.expand_dims(np.transpose(pickle.load(open('../Data/dna1', 'rb')), [0, 2, 1]), 3)
# dna2 = np.expand_dims(np.transpose(pickle.load(open('../Data/dna2', 'rb')), [0, 2, 1]), 3)
# print(dna1.shape)
#
# train = pd.read_csv('../Data/train.csv', encoding='utf-8')
# lab = train['label'].values.astype('int')
# lab = np.array(lab)
#
# # ------------build model & train-----------#
#
# def precision(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision
#
# slide1 = int(len(lab) * 0.6)
# slide2 = int(len(lab) * 0.8)
#
# model = buildModel()
# model.summary()
# model.compile(loss='binary_crossentropy',
#               optimizer=keras.optimizers.Adam(1e-3),
#               metrics=['accuracy',precision])
#
# ckpt = keras.callbacks.ModelCheckpoint('../Model/cba.h5',
#                                        monitor='val_precision',
#                                        verbose=1,
#                                        save_best_only=True,
#                                        save_weights_only=False,
#                                        mode='max',
#                                        period=1)
#
# history = model.fit([dna1[:slide1], dna2[:slide1]],
#                     lab[:slide1],
#                     epochs=25,
#                     batch_size=64,
#                     validation_data=([dna1[slide1:slide2], dna2[slide1:slide2]], lab[slide1:slide2]),
#                     callbacks=[ckpt])
#
# show_train_history(history.history, s='test', locate='../Result/')
# ROC(model, [dna1[slide1:], dna2[slide1:]], lab[slide1:])
#
# #--------------------predict----------------------------#
# test_data=[dna1[slide2:], dna2[slide2:]]
# test_label=lab[slide2:]
#
# rslt=model.predict()
# array=np.array([test_label,rslt[:,0]]).T
# df = pd.DataFrame(array, columns=['label', 'result'])
# df.to_csv('output.csv', index=False)

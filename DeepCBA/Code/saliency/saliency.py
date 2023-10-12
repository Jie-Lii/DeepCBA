# -*- coding: utf-8 -*-

'''
@ author LiJ
@ version 1.0
@ date 2023 / 10 / 11 20: 51
@ Description: visualize data by saliency
'''

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.scores import BinaryScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from data_encode import data_encode
from draw_heat_map import draw_heat_map

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


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


# loading data
file = pd.read_csv('../../../../../2023-08/2023-08-18/融合ppi和pdi的迁移学习/Data/B73/Dataset/B73(py)values_500_sin.csv', encoding='utf-8')
ann1 = file['Annotation1'].values.astype('str')[:10]
seq1 = file['Seq1'].values.astype('str')[:10]
ann2 = file['Annotation2'].values.astype('str')[:10]
seq2 = file['Seq2'].values.astype('str')[:10]
label = file['label'].values.astype('str')[:10]

# encoding data
print(' data_encoding '.center(100, '-') + '\n')
dna1 = data_encode(seq1)
dna2 = data_encode(seq2)
print(' encoding_success '.center(100, '$') + '\n')

# loading model
model = tf.keras.models.load_model('../../../../../2023-08/2023-08-18/融合ppi和pdi的迁移学习/Model/original_model/py.h5', custom_objects={'pearson': pearson})

'''
please change input_data format when model input is [dna1,dna2]
'''
#input_data = [dna1, dna2]
input_data = np.concatenate([dna1,dna2],axis=2)

# predicting data
pred_data = model.predict(input_data)

# visualizing data
for i in tqdm(range(len(label))):
    score = BinaryScore(label[i])
    replace2linear = ReplaceToLinear()
    saliency = Saliency(model, model_modifier=replace2linear, clone=True)

    '''
    please change input_data format when model input is [dna1,dna2]
    '''
    saliency_map = saliency(score, np.expand_dims(input_data[i], axis=[0, 3]), normalize_map=True)
    saliency_map = saliency_map[0]

    # drawing heat map
    draw_heat_map(saliency_map,'test','./')

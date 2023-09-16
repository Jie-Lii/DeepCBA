# -*- coding: utf-8 -*-
'''
@ author LiJ
@ version 1.0
@ date 2023 / 02 / 26 15: 11
@ Description: wheat exp data process
'''

import os
import Bio.SeqIO
import pandas as pd
import numpy as np
from tqdm import tqdm
from random import shuffle

wheat_tss_geneID = []
wheat_tts_geneID = []
# wheat_tss_seqs = []
# wheat_tts_seqs = []

for x in Bio.SeqIO.parse('../../../../小麦专用数据/序列数据/wheat_HC_gene_tss_1.5k_seq.fasta', 'fasta'):
    wheat_tss_geneID.append(x.id.split('_')[0])

for x in Bio.SeqIO.parse('../../../../小麦专用数据/序列数据/wheat_HC_gene_tts_1.5k_seq.fasta', 'fasta'):
    wheat_tts_geneID.append(x.id.split('_')[0])

file_name = os.listdir('../../../../小麦专用数据/交互数据/')
# file_name = file_name[::2]

for i in range(len(file_name)):
    wheat_int_data = pd.read_csv('../../../../小麦专用数据/交互数据/' + file_name[i], encoding='utf-8')
    wheat_int_data_ann1 = wheat_int_data['Gene_1'].values.astype('str')
    wheat_int_data_ann2 = wheat_int_data['Gene_2'].values.astype('str')
    wheat_int_data_exp1 = wheat_int_data['Gene_1.TPM'].values.astype('str')
    wheat_int_data_exp2 = wheat_int_data['Gene_2.TPM'].values.astype('str')

    for j in range(len(wheat_int_data_exp1)):
        if wheat_int_data_exp1[j]==' -   ':
            wheat_int_data_exp1[j]=0
        else:
            wheat_int_data_exp1[j]=float(wheat_int_data_exp1[j])

        if wheat_int_data_exp2[j]==' -   ':
            wheat_int_data_exp2[j]=0
        else:
            wheat_int_data_exp2[j]=float(wheat_int_data_exp2[j])
    wheat_int_data_exp1=wheat_int_data_exp1.astype('float')
    wheat_int_data_exp2=wheat_int_data_exp2.astype('float')

    save_wheat_data_ann1 = []
    save_wheat_data_ann2 = []
    save_wheat_data_exp = []

    for j in tqdm(range(len(wheat_int_data_ann1))):
        if wheat_int_data_ann1[j][-2:] == 'LC' or wheat_int_data_ann2[j][-2:] == 'LC':
            continue
        if wheat_int_data_ann1[j] not in wheat_tss_geneID or wheat_int_data_ann1[j] not in wheat_tts_geneID:
            continue
        if wheat_int_data_ann2[j] not in wheat_tss_geneID or wheat_int_data_ann2[j] not in wheat_tts_geneID:
            continue

        # data augument
        if  wheat_int_data_exp2[j] <= 500:
            save_wheat_data_ann1.append(wheat_int_data_ann1[j])
            save_wheat_data_ann2.append(wheat_int_data_ann2[j])
            save_wheat_data_exp.append(wheat_int_data_exp2[j])

        if wheat_int_data_exp1[j] <= 500:
            save_wheat_data_ann1.append(wheat_int_data_ann2[j])
            save_wheat_data_ann2.append(wheat_int_data_ann1[j])
            save_wheat_data_exp.append(wheat_int_data_exp1[j])

    tmp = list(zip(save_wheat_data_ann1, save_wheat_data_ann2, save_wheat_data_exp))
    shuffle(tmp)
    save_wheat_data_ann1 = [j[0] for j in tmp]
    save_wheat_data_ann2 = [j[1] for j in tmp]
    save_wheat_data_exp = [j[2] for j in tmp]

    df = pd.DataFrame(
        {'Annotation1': save_wheat_data_ann1, 'Annotation2': save_wheat_data_ann2, 'exp': save_wheat_data_exp})
    df.to_csv('../Data/小麦/' + file_name[i], mode='w', header=True, index=None)

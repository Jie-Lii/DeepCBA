# -*- coding: utf-8 -*-
'''
@ author LiJ
@ version 1.0
@ date 2023 / 02 / 26 15: 11
@ Description: rice exp data process
'''

import os
import Bio.SeqIO
import pandas as pd
import numpy as np
from tqdm import tqdm
from random import shuffle

rice_exp_data = pd.read_csv('../../../../水稻专用数据/表达量数据/ZS_Expression_data.csv', encoding='utf-8')
rice_exp_data_gene = rice_exp_data['Gene'].values.astype('str').tolist()
rice_exp_data_exp = rice_exp_data['Expression'].values.astype('float').tolist()

rice_tss_geneID = []
rice_tts_geneID = []
#rice_tss_seqs = []
#rice_tts_seqs = []

for x in Bio.SeqIO.parse('../../../../水稻专用数据/序列数据/ZS_gene_tss_1.5kb_seq_after_rename.fasta', 'fasta'):
    rice_tss_geneID.append(x.id.split('_')[0])
    #rice_tss_seqs.append(x.seq)

for x in Bio.SeqIO.parse('../../../../水稻专用数据/序列数据/ZS_gene_tts_1.5kb_seq_after_rename.fasta', 'fasta'):
    rice_tts_geneID.append(x.id.split('_')[0])
    #rice_tts_seqs.append(x.seq)

file_name = os.listdir('../../../../水稻专用数据/交互数据/ZS/正样本/')
#file_name = file_name[::2]

for i in range(len(file_name)):
    rice_int_data = pd.read_csv('../../../../水稻专用数据/交互数据/ZS/正样本/' + file_name[i], encoding='utf-8')
    rice_int_data_ann1 = rice_int_data['Annotation1'].values.astype('str')
    rice_int_data_ann2 = rice_int_data['Annotation2'].values.astype('str')

    save_rice_data_ann1 = []
    save_rice_data_ann2 = []
    save_rice_data_exp=[]

    for j in tqdm(range(len(rice_int_data_ann1))):
        if rice_int_data_ann1[j] == 'NA' or rice_int_data_ann2[j] == 'NA':
            continue

        if rice_int_data_ann1[j] not in rice_tss_geneID or rice_int_data_ann1[j] not in rice_tts_geneID:
            continue
        if rice_int_data_ann2[j] not in rice_tss_geneID or rice_int_data_ann2[j] not in rice_tts_geneID:
            continue

        if rice_int_data_ann1[j] not in rice_exp_data_gene or rice_int_data_ann2[j] not in rice_exp_data_gene:
            continue

        # data augument
        if rice_exp_data_exp[rice_exp_data_gene.index(rice_int_data_ann2[j])]<=500 :
            save_rice_data_ann1.append(rice_int_data_ann1[j])
            save_rice_data_ann2.append(rice_int_data_ann2[j])
            save_rice_data_exp.append(rice_exp_data_exp[rice_exp_data_gene.index(rice_int_data_ann2[j])])

        if rice_exp_data_exp[rice_exp_data_gene.index(rice_int_data_ann1[j])]<=500 :
            save_rice_data_ann1.append(rice_int_data_ann2[j])
            save_rice_data_ann2.append(rice_int_data_ann1[j])
            save_rice_data_exp.append(rice_exp_data_exp[rice_exp_data_gene.index(rice_int_data_ann1[j])])

    tmp = list(zip(save_rice_data_ann1,save_rice_data_ann2,save_rice_data_exp))
    shuffle(tmp)
    save_rice_data_ann1 = [j[0] for j in tmp]
    save_rice_data_ann2 = [j[1] for j in tmp]
    save_rice_data_exp= [j[2] for j in tmp]

    df = pd.DataFrame({'Annotation1': save_rice_data_ann1, 'Annotation2': save_rice_data_ann2, 'exp': save_rice_data_exp})
    df.to_csv('../Data/水稻/ZS_'+file_name[i], mode='w', header=True,index=None)
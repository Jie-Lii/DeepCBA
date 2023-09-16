# -*- coding: utf-8 -*-

'''
@ author LiJ
@ version 1.0
@ date 2023 / 08 / 18 20: 16
@ Description: build data set for maize
'''
import os
import Bio.SeqIO
import numpy as np
import pandas as pd
from tqdm import tqdm


def read_fasta_file():
    geneID = []
    seqs = []
    for x in Bio.SeqIO.parse('../../../../玉米专用数据/玉米B73_PDI_SeqData_len=1500/PDI_1.5k_seq.fasta', 'fasta'):
        if str(x.id)[:2]=='Zm':
            geneID.append(str(x.id).split('_')[0])
            seqs.append(str(x.seq))
        elif str(x.id)[:2]=='Dm':
            geneID.append(str(x.id).split(':')[0])
            seqs.append(str(x.seq))
    seqs_dict = dict(zip(geneID, seqs))
    return seqs_dict


def build_B73_Data_Set(open_file_name):
    file = pd.read_csv('../Data/B73/%s' % open_file_name, encoding='utf-8')
    ann1 = file['Annotation1'].values.astype('str')
    ann2 = file['Annotation2'].values.astype('str')
    exp2 = file['express'].values.astype('float')

    seqs_dict = read_fasta_file()

    ann1_name, ann2_name, seqs1, seqs2, exp = [], [], [], [], []

    for i in tqdm(range(len(ann1))):
        if (ann1[i] in seqs_dict) and (ann2[i] in seqs_dict):
            ann1_name.append(ann1[i])
            ann2_name.append(ann2[i])
            seqs1.append(seqs_dict[ann1[i]])
            seqs2.append(seqs_dict[ann2[i]])
            exp.append(exp2[i])

    out_data = np.array([ann1_name, seqs1, ann2_name, seqs2, exp]).T
    np.random.shuffle(out_data)
    df = pd.DataFrame(out_data,
                      columns=['Annotation1', 'Seq1',
                               'Annotation2', 'Seq2', 'label'])
    df.to_csv('../Data/B73/Dataset/%s'%open_file_name, index=False)

files=os.listdir('../Data/B73/')
for file in files:
    build_B73_Data_Set(file)

# -*- coding: utf-8 -*-

'''
@ author LiJ
@ version 1.0
@ date 2023 / 04 / 09 10: 00
@ Description: process cotton exp data
'''

import os
import Bio.SeqIO
import numpy as np
import pandas as pd
from tqdm import tqdm

exp_file = pd.read_csv('../Data/Expression-data/Cotton_all_gene_FPKM_expression.csv', encoding='utf-8')
exp_gene = exp_file['Gene'].values.astype('str')
exp = exp_file['Expression'].values.astype('float')
exp_dict = dict(zip(exp_gene, exp))

def process_cotton_data(open_file_path):
    xlsx_files = os.listdir('../Data/Interaction-data/%s' % (open_file_path))

    en_name = []
    en_seq = []
    ge_name = []
    ge_seq = []
    for x in Bio.SeqIO.parse('../Data/Seq-data/%s_enhancer.fa' % (open_file_path), 'fasta'):
        en_name.append(x.id.split(':')[0])
        en_seq.append(str(x.seq))
    for x in Bio.SeqIO.parse('../Data/Seq-data/%s_gene.fasta' % (open_file_path), 'fasta'):
        ge_name.append(x.id.split(':')[0])
        ge_seq.append(str(x.seq))
    en_dict = dict(zip(en_name, en_seq))
    ge_dict = dict(zip(ge_name, ge_seq))

    pros_file = pd.read_excel('../Data/Interaction-data/%s/gene-genedu.xlsx' % (open_file_path))
    pros_file = pros_file.values.astype('str')
    ann1 = pros_file[:, 3].astype('str')
    ann2 = pros_file[:, 11].astype('str')

    out_ann1 = []
    out_ann2 = []
    out_seq1 = []
    out_seq2 = []
    out_exp = []

    for i in tqdm(range(len(ann1))):
        if (ann2[i] in exp_dict) and (ann1[i] in ge_dict) and (ann2[i] in ge_dict) and (exp_dict[ann2[i]]<=500):
            out_ann1.append(ann1[i])
            out_ann2.append(ann2[i])
            out_seq1.append(ge_dict[ann1[i]])
            out_seq2.append(ge_dict[ann2[i]])
            out_exp.append(exp_dict[ann2[i]])
    df = pd.DataFrame([out_ann1, out_seq1, out_ann2, out_seq2, out_exp]).T
    df.columns = ['Annotation1', 'Seq1', 'Annotation2', 'Seq2', 'Expression']
    df.to_csv('../Data/%s_gene-gene_dataset.csv' % (open_file_path), index=False)

    pros_file = pd.read_excel('../Data/Interaction-data/%s/gene-enhancerdu.xlsx' % (open_file_path))
    pros_file = pros_file.values.astype('str')
    ann1 = pros_file[:, 3].astype('str')
    ann2 = pros_file[:, 11].astype('str')

    out_ann1 = []
    out_ann2 = []
    out_seq1 = []
    out_seq2 = []
    out_exp = []

    for i in tqdm(range(len(ann1))):
        if (ann1[i] in exp_dict) and (ann1[i] in ge_dict) and (ann2[i] in en_dict) and (exp_dict[ann1[i]]<=500):
            out_ann1.append(ann1[i])
            out_ann2.append(ann2[i])
            out_seq1.append(ge_dict[ann1[i]])
            out_seq2.append(en_dict[ann2[i]])
            out_exp.append(exp_dict[ann1[i]])
    df = pd.DataFrame([out_ann1, out_seq1, out_ann2, out_seq2, out_exp]).T
    df.columns = ['Annotation1', 'Seq1', 'Annotation2', 'Seq2', 'Expression']
    df.to_csv('../Data/%s_gene-enhancer_dataset.csv' % (open_file_path), index=False)


    pros_file = pd.read_excel('../Data/Interaction-data/%s/enhancer-genedu.xlsx' % (open_file_path))
    pros_file = pros_file.values.astype('str')
    ann1 = pros_file[:, 3].astype('str')
    ann2 = pros_file[:, 8].astype('str')

    out_ann1 = []
    out_ann2 = []
    out_seq1 = []
    out_seq2 = []
    out_exp = []

    for i in tqdm(range(len(ann1))):
        if (ann2[i] in exp_dict) and (ann1[i] in en_dict) and (ann2[i] in ge_dict) and (exp_dict[ann2[i]]<=500):
            out_ann1.append(ann1[i])
            out_ann2.append(ann2[i])
            out_seq1.append(en_dict[ann1[i]])
            out_seq2.append(ge_dict[ann2[i]])
            out_exp.append(exp_dict[ann2[i]])
    df = pd.DataFrame([out_ann1, out_seq1, out_ann2, out_seq2, out_exp]).T
    df.columns = ['Annotation1', 'Seq1', 'Annotation2', 'Seq2', 'Expression']
    df.to_csv('../Data/%s_enhancer-gene_dataset.csv' % (open_file_path), index=False)


files=os.listdir('../Data/Interaction-data/')
for file in files:
    process_cotton_data(file)

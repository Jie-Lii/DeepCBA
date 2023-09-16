# -*- coding: utf-8 -*-
'''
@ author LiJ
@ version 1.0
@ date 2023 / 03 / 21 18: 46
@ Description: data encode
'''

import os
import pickle
import pandas as pd
import numpy as np
import Bio.SeqIO
from tqdm import tqdm


def data_encode(seqs):
    dna = np.zeros(shape=(len(seqs), len(seqs[0]), 4), dtype='float16')  # shape=(x,4,3000)
    for i in tqdm(range(len(seqs))):
        tmp_seq = seqs[i]
        for j in range(len(seqs[i])):
            if tmp_seq[j] == 'A':
                dna[i][j][0] = 1
            elif tmp_seq[j] == 'C':
                dna[i][j][1] = 1
            elif tmp_seq[j] == 'G':
                dna[i][j][2] = 1
            elif tmp_seq[j] == 'T':
                dna[i][j][3] = 1
            else:
                pass
    dna = np.transpose(dna, axes=[0, 2, 1])
    return dna
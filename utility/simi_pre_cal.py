#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Calculate the sample pairwise distance matrix S_l under feature l

@Project ：UFS_methods
@File ：simi_pre_cal.py
@Author ：WANG CC
"""

from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

from utility.unsupervised_evaluation import dataset_pro, dataset_info

Data_names = ['Yale', 'LUNG', 'lung_discrete', 'TOX_171']

for data_name in ['sEMG']:
    print("data name:", data_name)
    X, Y, Classes = dataset_pro(data_name, '')
    N_samples, n_features = X.shape
    LABEL, CLASS, LABEL_INFO = dataset_info(data_name, '-')
    print(N_samples, n_features, CLASS)
    print(LABEL_INFO)
    sl_all = []
    for ik in range(n_features):
        sl_all.append(pairwise_distances(X[:, ik].reshape(N_samples, 1)) ** 2)
    sl_all = np.array(sl_all)
    np.save('D:/Exp/Exp-Datasets/simi_pre/' + data_name + '.npy', sl_all)

import numpy as np
from numpy import linalg as LA
from decimal import *
eps = np.spacing(1)


def feature_ranking(w):
    """
    This function ranks features according to the feature weights matrix W

    Input:
    -----
    W: {numpy array}, shape (n_features, n_classes)
        feature weights matrix

    Output:
    ------
    idx: {numpy array}, shape {n_features,}
        feature index ranked in descending order by feature importance
    """
    t = (w * w).sum(1)
    idx = np.argsort(t, 0)
    return idx[::-1]


def generate_diagonal_matrix(u):
    """
    This function generates a diagonal matrix D from an input matrix U as D_ii = 0.5 / ||U[i,:]||

    Input:
    -----
    U: {numpy array}, shape (n_samples, n_features)

    Output:
    ------
    D: {numpy array}, shape (n_samples, n_samples)
    """
    temp = np.sqrt((u * u).sum(1))  # 对应元素相乘按行相加
    temp[temp < 1e-16] = 1e-16
    temp = 0.5 / temp
    d = np.diag(temp)
    return d


def generate_diagonal_matrix_l2p(u, p):
    """
    This function generates a diagonal matrix D from an input matrix U as D_ii = 0.5 / ||U[i,:]||

    Input:
    -----
    U: {numpy array}, shape (n_samples, n_features)

    Output:
    ------
    D: {numpy array}, shape (n_samples, n_samples)
    """
    temp = (u*u).sum(1)  # 对应元素相乘按行相加
    temp = np.power(temp+1e-16, (p-2) / 2)
    d = np.diag(temp) * p / 2
    return d


def calculate_l21_norm(x):
    """
    This function calculates the l21 norm of a matrix X, i.e., sum ||X[i,:]||_2

    Input:
    -----
    X: {numpy array}, shape (n_samples, n_features)

    Output:
    ------
    l21_norm: {float}
    """
    return (np.sqrt(np.multiply(x, x).sum(1))).sum()


def calculate_l2p_norm(w, p):
    """
    :param w: d,c
    :param p: 0<p<=1
    :return: ||w||_{2,p}^p
    """
    return np.sum(np.power((w * w).sum(1), p / 2))


def construct_label_matrix(label, methods):
    """
    This function converts a 1d numpy array to a 2d array, for each instance, the class label is 1 or 0

    Input:
    -----
    label: {numpy array}, shape(n_samples,)

    Output:
    ------
    label_matrix: {numpy array}, shape(n_samples, n_classes)
    """

    n_samples = label.shape[0]
    import pandas as pd
    class_num = pd.DataFrame(label).value_counts()  # 计算不同类样本出现的个数
    unique_label = np.unique(label)  # 去除重复的数字后排序
    n_classes = unique_label.shape[0]

    label_matrix = np.zeros((n_samples, n_classes))
    if methods == 're_sqrt':
        for i in range(n_classes):
            label_matrix[label == unique_label[i], i] = 1 / np.sqrt(class_num[i])
        return label_matrix
    else:
        for i in range(n_classes):
            label_matrix[label == unique_label[i], i] = 1
        return label_matrix.astype(int)


def construct_label_matrix_pan(label):
    """
    This function converts a 1d numpy array to a 2d array, for each instance, the class label is 1 or -1

    Input:
    -----
    label: {numpy array}, shape(n_samples,)

    Output:
    ------
    label_matrix: {numpy array}, shape(n_samples, n_classes)
    """
    n_samples = label.shape[0]
    unique_label = np.unique(label)
    n_classes = unique_label.shape[0]
    label_matrix = np.zeros((n_samples, n_classes))
    for i in range(n_classes):
        label_matrix[label == unique_label[i], i] = 1
    label_matrix[label_matrix == 0] = -1

    return label_matrix.astype(int)


def euclidean_projection(v, n_features, n_classes, z, gamma):
    """
    L2 Norm regularized euclidean projection min_W  1/2 ||W- V||_2^2 + z * ||W||_2
    """
    w_projection = np.zeros((n_features, n_classes))
    for i in range(n_features):
        if LA.norm(v[i, :]) > z / gamma:
            w_projection[i, :] = (1 - z / (gamma * LA.norm(v[i, :]))) * v[i, :]
        else:
            w_projection[i, :] = np.zeros(n_classes)
    return w_projection


def tree_lasso_projection(v, n_features, idx, n_nodes):
    """
    This functions solves the following optimization problem min_w 1/2 ||w-v||_2^2 + sum z_i||w_{G_{i}}||
    where w and v are of dimensions of n_features; z_i >=0, and G_{i} follows the tree structure
    """
    # test whether the first node is special
    if idx[0, 0] == -1 and idx[1, 0] == -1:
        w_projection = np.zeros(n_features)
        z = idx[2, 0]
        for j in range(n_features):
            if v[j] > z:
                w_projection[j] = v[j] - z
            else:
                if v[j] < -z:
                    w_projection[j] = v[j] + z
                else:
                    w_projection[j] = 0
        i = 1

    else:
        w = v.copy()
        i = 0

    # sequentially process each node
    while i < n_nodes:
        # compute the L2 norm of this group
        two_norm = 0
        start_idx = int(idx[0, i] - 1)
        end_idx = int(idx[1, i])
        for j in range(start_idx, end_idx):
            two_norm += w_projection[j] * w_projection[j]
        two_norm = np.sqrt(two_norm)
        z = idx[2, i]
        if two_norm > z:
            ratio = (two_norm - z) / two_norm
            # shrinkage this group by ratio
            for j in range(start_idx, end_idx):
                w_projection[j] *= ratio
        else:
            for j in range(start_idx, end_idx):
                w_projection[j] = 0
        i += 1
    return w_projection


def tree_norm(w, n_features, idx, n_nodes):
    """
    This function computes sum z_i||w_{G_{i}}||
    """
    obj = 0
    # test whether the first node is special
    if idx[0, 0] == -1 and idx[1, 0] == -1:
        z = idx[2, 0]
        for j in range(n_features):
            obj += np.abs(w[j])
        obj *= z
        i = 1
    else:
        i = 0

    # sequentially process each node
    while i < n_nodes:
        two_norm = 0
        start_idx = int(idx[0, i] - 1)
        end_idx = int(idx[1, i])
        for j in range(start_idx, end_idx):
            two_norm += w[j] * w[j]
        two_norm = np.sqrt(two_norm)
        z = idx[2, i]
        obj += z * two_norm
        i += 1
    return obj

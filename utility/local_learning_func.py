#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：UFS_methods 
@File ：local_learning_func.py
@Author ：WANG CC
'''
import numpy as np
eps = np.spacing(1)


def eig_lastk(a, k):
    """
    the k smallest eig_vectors and eig_values
    :param a: input matrix
    :param k: number of eig_vectors
    :return:
    """
    a = np.maximum(a, a.T)
    e_vals, e_vecs = np.linalg.eig(a)
    e_vals = np.real(e_vals)
    e_vecs = np.real(e_vecs)
    sorted_indices = np.argsort(e_vals)
    return e_vecs[:, sorted_indices[:k]], e_vals[sorted_indices]  # e_vals[sorted_indices[:k]]


def estimateReg(distx, k):
    """
    learning \mu in adaptive local learning
    :param distx: sample pairwise distance
    :param k: the neighbors
    :return: initial similarity matrix
    """
    n_sample = distx.shape[0]
    idx = np.argsort(distx)
    distx1 = np.sort(distx)
    a = np.zeros((n_sample, n_sample))
    rr = np.zeros((n_sample, 1))
    for i in range(n_sample):
        di = distx1[i, 1:k + 2]
        rr[i] = 0.5 * (k * di[k] - sum(di[:k]))
        id_ = idx[i, 1:k + 2]
        a[i, id_] = (di[k] - di) / (k * di[k] - sum(di[:k]) + eps)
    r = np.mean(rr)
    return r, a


def EProjSimplex_new(v, k):
    """
        min  1/2 || x - v||^2
        s.t. x>=0, 1'x=1
    :param v:
    :param k:
    :return: x, ft
    """
    ft = 1
    n = v.shape[0]
    v1 = np.zeros(n)
    v0 = v - np.mean(v) + k / n
    vmin = min(v0)
    if vmin < 0:
        f = 1
        lambda_m = 0
        while abs(f) > 10e-10:
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = sum(posidx)
            g = -npos
            f = np.sum(v1[posidx]) - k  # sum the pos element in v1
            lambda_m = lambda_m - f / g
            ft += 1
            if ft > 100:
                x = np.maximum(v1, 0)
                return x, ft
        x = np.maximum(v1, 0)
    else:
        x = v0
    return x


def local_structure_learning(k, alpha, dist_x, dist_f, islocal):
    """
    min_s ||x-x||s + alpha * S
    s.t. s1=1, s>=0.
    :param islocal: True with only k neighbors, False with all neighbors
    :param dist_f:
    :param dist_x:
    :param alpha:  learned by local_reg = estimateReg(x, k)
    :param k: k neighbors
    :return : sym s
    """
    n_sample = dist_x.shape[0]
    s = np.zeros((n_sample, n_sample))
    # local_reg = estimateReg(x, k)
    idx = np.argsort(dist_x)  # 每行排序
    for i_smp in range(n_sample):
        if islocal:
            idx_a0 = idx[i_smp, 1:k + 1]
        else:
            idx_a0 = np.arange(n_sample)
        dxi = dist_x[i_smp, idx_a0]
        dfi = dist_f[i_smp, idx_a0]
        ad = - (dxi + dfi) / (2 * alpha)
        s[i_smp, idx_a0] = EProjSimplex_new(ad, 1)
    s = (s + s.T) / 2
    return s

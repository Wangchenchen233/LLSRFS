# -*- coding: utf-8 -*-
# @Time    : 2021/12/17 9:38
# @Author  : WANG CC
# @Email   : wangchenchen233@163.com
# @File    : 1217_final_all.py

from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from utility.local_learning_func import local_structure_learning, eig_lastk
eps = np.spacing(1)


def LLSRFS(x, c, para, s0, f0, local_reg, sl_all):
    """
    XWP-F + P_21

    :param local_reg:
    :param f0:
    :param s0:
    :param para:
    :param x: d n
    :param c: cluster number
    :return: w
    """
    para_b, lamb_p, lamb_w = para
    # beta = 1
    n_feature, n_sample = x.shape
    h = np.eye(n_sample)-np.ones((n_sample, n_sample))/n_sample
    xxt = np.dot(np.dot(x, h), x.T)
    s = s0
    f = f0
    k = 5
    # init ls
    lamb_f = local_reg
    # init w
    w_v = np.ones(n_feature) / n_feature
    w_m = np.diag(w_v)
    max_iter = 100
    e_val_all = []
    w_v_iter = []
    for iter_ in range(max_iter):
        # calculate p
        w_ = np.linalg.inv(w_m)
        p_inv = np.linalg.inv(xxt + lamb_w * w_ * w_)
        p_x = np.dot(p_inv, x)
        p = np.dot(np.dot(p_x, h), f)

        # calculate w
        # upper
        w_u = np.power((p * p).sum(1) + eps, 1/(2*para_b+1))
        dl = np.zeros(n_feature)
        for k in range(n_feature):
            dl[k] = np.sum(sl_all[k] * s)
            # low
            w_l = np.power((dl[k] + eps), 1/(2*para_b+1))
            w_v[k] = w_u[k] / w_l

        # w_v_temp = np.exp(-pl / lambda_w)
        w_v /= np.sum(w_v)
        w_m = np.diag(np.power(w_v, para_b))

        # calculate s
        x2 = np.dot(np.diag(np.sqrt(w_v)), x)
        dist_x = pairwise_distances(x2.T) ** 2
        f_old = f
        dist_f = lamb_f * pairwise_distances(f) ** 2
        s = local_structure_learning(k, local_reg, dist_x, dist_f, 0)
        ls = np.diag(s.sum(0)) - s

        # update F
        lm = lamb_p * np.dot(np.dot(h, (np.eye(n_sample) - np.dot(x.T, p_x))), h) + lamb_f * ls
        f, _ = eig_lastk(lm, c)
        _, e_val = eig_lastk(ls, c)
        e_val_all.append(e_val)

        fn1 = np.sum(e_val[:c])
        fn2 = np.sum(e_val[:c + 1])

        if fn1 > 10e-10:
            lamb_f *= 2
        elif fn2 < 10e-10:
            lamb_f /= 2
            f = f_old
        else:
            break
    return w_v, s

# -*- coding: utf-8 -*-
# @Time    : 2022/1/10 15:47
# @Author  : WANG CC
# @Email   : wangchenchen233@163.com
# @File    : SRFS.py
# -*- coding: utf-8 -*-
# @Time    : 2021/12/17 9:38
# @Author  : WANG CC
# @Email   : wangchenchen233@163.com
# @File    : 1217_final_all.py

import numpy as np
from utility.local_learning_func import eig_lastk
eps = np.spacing(1)


def SRFS(x, c, para, ls):
    """
    XWP-F + P_21
    :param f0:
    :param ls:
    :param para:
    :param x: d n
    :param c: cluster number
    :return: w
    """
    lamb_q, lamb_f, lamb_w = para
    # beta = 1
    n_feature, n_sample = x.shape
    h = np.eye(n_sample)  # - np.ones((n_sample, n_sample)) / n_sample
    xh = np.dot(x, h)
    xxt = np.dot(xh, x.T)
    # init w
    w_v = np.ones(n_feature) / n_feature
    w_ = np.eye(n_feature)
    max_iter = 30
    w_v_old = np.zeros(n_feature)
    obj = np.zeros(max_iter)
    for iter_ in range(max_iter):
        # calculate p
        p_inv = np.linalg.inv(xxt + lamb_w * w_ * w_)
        p_x = np.dot(p_inv, xh)

        # update F
        lm = h - np.dot(xh.T, p_x) + lamb_f * ls
        f, _ = eig_lastk(lm, c)
        f = np.array(f)

        p = np.dot(p_x, f)

        # calculate w
        w_v_old = w_v
        w_v = np.power((p * p).sum(1), 1 / (2 * lamb_q + 1))
        w_v /= np.sum(w_v)
        w_v_q = np.power(w_v, lamb_q)
        w_v_q[w_v_q < 1e-16] = 1e-16
        w_ = np.diag(1 / w_v_q)

        p_norm = (p * p).sum(1)
        obj[iter_] = np.linalg.norm(
            np.dot(h, np.dot(x.T, p)) - np.dot(h, f), ord='f') ** 2 + lamb_w * np.sum(
            p_norm / np.power(w_v_q, 2)) + lamb_f * np.trace(
            np.dot(np.dot(f.T, ls), f))

        print('obj' + str(iter_ + 1) + ': ' + str(obj[iter_]))

        if iter_ >= 1 and (abs(obj[iter_] - obj[iter_ - 1]) < 1e-3):
            break
    return w_v

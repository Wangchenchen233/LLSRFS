# -*- coding: utf-8 -*-
# @Time    : 2021/12/7 14:18
# @Author  : WANG CC
# @Email   : wangchenchen233@163.com
# @File    : graph_embedding_multi_class.py
import numpy as np
import sklearn
import SOGFS
import LLSRFS
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from scipy.sparse import *
from utility.local_learning_func import estimateReg, eig_lastk
eps = np.spacing(1)


def generate_multi_class(n):
    np.random.seed(223)
    """
    generate n*4
    :param n:
    :return:
    """
    # generate the first class
    x1 = np.random.multivariate_normal([1, 1], np.diag([0.05, 0.05]), n).T  # (2,100) gaussian distribution
    x1 = np.vstack((x1, np.random.rand(n) * 5))  # [0,5] uniform distribution
    x1 = np.vstack((x1, np.random.rand(n) * 3 + 1))  # [1,4] uniform distribution
    y1 = np.ones(n)

    # generate the second class
    x2 = np.random.multivariate_normal([2, 1], np.diag([0.05, 0.05]), n).T  # (2,100) gaussian distribution
    x2 = np.vstack((x2, np.random.rand(n) * 5))  # [0,5]
    x2 = np.vstack((x2, np.random.rand(n) * 3 + 1))  # [1,4]
    y2 = np.ones(n) * 2

    # generate the third class
    x3 = np.random.multivariate_normal([1, 3], np.diag([0.6, 0.05]), n).T  # (2,100) gaussian distribution
    x3 = np.vstack((x3, np.random.rand(n) * 5))  # [0,5]
    x3 = np.vstack((x3, np.random.rand(n) * 3 + 1))  # [1,4]
    y3 = np.ones(n) * 3
    x12 = np.hstack((x1, x2))
    y12 = np.hstack((y1, y2))
    x = np.hstack((x12, x3)).T  # (n, d)
    y = np.hstack((y12, y3)).T  # (n, )
    """
    plt.subplot(231)
    plt.plot(x1[0, :], x1[1, :], '.r', alpha=0.6)
    plt.plot(x2[0, :], x2[1, :], '.b')
    plt.plot(x3[0, :], x3[1, :], '.g')
    plt.xlabel('f1')
    plt.ylabel('f2')

    plt.subplot(232)
    plt.plot(x1[0, :], x1[2, :], '.r')
    plt.plot(x2[0, :], x2[2, :], '.b')
    plt.plot(x3[0, :], x3[2, :], '.g')
    plt.xlabel('f1')
    plt.ylabel('f3')

    plt.subplot(233)
    plt.plot(x1[0, :], x1[3, :], '.r')
    plt.plot(x2[0, :], x2[3, :], '.b')
    plt.plot(x3[0, :], x3[3, :], '.g')
    plt.xlabel('f1')
    plt.ylabel('f4')

    plt.subplot(235)
    plt.plot(x1[1, :], x1[2, :], '.r')
    plt.plot(x2[1, :], x2[2, :], '.b')
    plt.plot(x3[1, :], x3[2, :], '.g')
    plt.xlabel('f2')
    plt.ylabel('f3')

    plt.subplot(236)
    plt.plot(x1[1, :], x1[3, :], '.r')
    plt.plot(x2[1, :], x2[3, :], '.b')
    plt.plot(x3[1, :], x3[3, :], '.g')
    plt.xlabel('f2')
    plt.ylabel('f4')

    plt.subplot(234)
    plt.plot(x1[2, :], x1[3, :], '.r')
    plt.plot(x2[2, :], x2[3, :], '.b')
    plt.plot(x3[2, :], x3[3, :], '.g')
    plt.xlabel('f3')
    plt.ylabel('f4')
    plt.show()
"""
    return x, y


def similar_matrix(x, k, t_c):
    """
    :param t_c: scale for para t
    :param x: N D
    :param k:
    :return:
    """
    # compute pairwise euclidean distances
    n_samples, n_features = x.shape
    D = pairwise_distances(x)
    D **= 2
    # sort the distance matrix D in ascending order
    dump = np.sort(D, axis=1)
    idx = np.argsort(D, axis=1)
    # 0值:沿着每一列索引值向下执行方法(axis=0代表往跨行)分别对每一列
    # 1值:沿着每一行(axis=1代表跨列) 分别对每一行
    idx_new = idx[:, 0:k + 1]
    dump_new = dump[:, 0:k + 1]
    # compute the pairwise heat kernel distances
    # t = np.percentile(D.flatten(), 20)  # 20210816 tkde13
    t = np.mean(D)
    t = t_c * t
    dump_heat_kernel = np.exp(-dump_new / (2 * t))
    G = np.zeros((n_samples * (k + 1), 3))
    G[:, 0] = np.tile(np.arange(n_samples), (k + 1, 1)).reshape(-1)  # 第一个参数为Y轴扩大倍数，第二个为X轴扩大倍数
    G[:, 1] = np.ravel(idx_new, order='F')  # 按列顺序重塑 n_samples*(k+1)
    G[:, 2] = np.ravel(dump_heat_kernel, order='F')
    # build the sparse affinity matrix W
    W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
    bigger = np.transpose(W) > W
    W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
    # np.transpose(W).multiply(bigger)不等于np.multiply(W,bigger)
    return W


def generate_one_gaussian_class(mu, sig, n):
    """
    generate one gaussian class with given mean and sigma
    :param mu:
    :param sig:
    :param n:
    :return:
    """
    # n,2
    x = np.random.multivariate_normal(mu, sig, n).T
    x = np.vstack((x, np.random.rand(n) * 5))  # [0,5] uniform distribution
    x = np.vstack((x, np.random.rand(n) * 3 + 1))  # [1,4] uniform distribution
    y = np.ones(n)
    return x, y


if __name__ == '__main__':
    X, y = generate_multi_class(100)
    X = sklearn.preprocessing.scale(X)

    fig = plt.figure(figsize=(16, 5))
    plt.subplot(1, 3, 1)
    plt.plot(X[:100, 0], X[:100, 1], '.r')
    plt.plot(X[100:200, 0], X[100:200, 1], '.b')
    plt.plot(X[200:300, 0], X[200:300, 1], '.g')
    plt.text(-0.7, -2.25, "(a) Real Distribution",
             horizontalalignment='center',
             fontsize=14)
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.tight_layout()

    classes = 3
    S0 = similar_matrix(X, 5, 1)
    Dist_x = pairwise_distances(X) ** 2
    Local_reg, S = estimateReg(Dist_x, 5)
    S = (S + S.T) / 2
    Ls = np.diag(S.sum(0)) - S
    F, E_val = eig_lastk(Ls, classes)
    if np.sum(E_val[:classes + 1]) < 10e-11:
        print("already c connected component")
    W, S1 = SOGFS.SOGFS(X, 3, (1, 0.1, 2), Ls, F, Local_reg)
    X_W_s = np.dot(X, W)

    plt.subplot(1, 3, 2)
    plt.plot(X_W_s[:100, 0], X_W_s[:100, 1], '.r')
    plt.plot(X_W_s[100:200, 0], X_W_s[100:200, 1], '.b')
    plt.plot(X_W_s[200:300, 0], X_W_s[200:300, 1], '.g')
    # plt.title("(b) Low-dimensional Embedding Space", y=-0.1)
    plt.text(0, -3, "(b) Low-dimensional Space(SOGFS)",
             horizontalalignment='center',
             fontsize=14)
    plt.xlabel('first dimension', fontsize=14)
    plt.ylabel('second dimension', fontsize=14)
    plt.tight_layout()


    sl_all = []
    for ik in range(4):
        sl_all.append(pairwise_distances(X[:, ik].reshape(300, 1)) ** 2)
    sl_all = np.array(sl_all)
    w_v, S2 = LLSRFS.LLSRFS(X.T, 3, (0.5, 1, 1), S, F, Local_reg, sl_all)
    X_W = np.dot(X, np.diag(w_v))

    plt.subplot(1, 3, 3)
    plt.plot(X_W[:100, 0], X_W[:100, 1], '.r')
    plt.plot(X_W[100:200, 0], X_W[100:200, 1], '.b')
    plt.plot(X_W[200:300, 0], X_W[200:300, 1], '.g')
    # plt.title("(c) Feature Subspace", y=-0.1)
    plt.text(-0.25, -1.25, "(c) Feature Subspace(Ours)",
             horizontalalignment='center',
             fontsize=14)
    plt.xlabel("feature 1", fontsize=14)
    plt.ylabel("feature 2", fontsize=14)
    plt.tight_layout()
    plt.show()


    import seaborn as sns
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 3, 1)
    sns.heatmap(S0.toarray(), cmap=sns.cm.rocket_r)
    plt.text(150, 350, "(a) Original Space($k$-nn)",
             horizontalalignment='center',
             fontsize=14)
    plt.tight_layout()
    plt.subplot(1, 3, 2)
    sns.heatmap(S1, cmap=sns.cm.rocket_r, vmin=0, vmax=0.3)
    plt.text(150, 350, "(b) Low-dimensional Space(SOGFS)",
             horizontalalignment='center',
             fontsize=14)
    plt.tight_layout()
    plt.subplot(1, 3, 3)
    sns.heatmap(S2, cmap=sns.cm.rocket_r, vmin=0, vmax=0.3)
    plt.text(150, 350, "(c) Feature Subspace(Ours)",
             horizontalalignment='center',
             fontsize=14)
    plt.tight_layout()
    plt.show()

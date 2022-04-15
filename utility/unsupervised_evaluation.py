import numpy as np
# import sklearn.utils.linear_assignment_ as la
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
import xlwt
import scipy.io as scio
import sklearn
import pandas as pd
import os
from tqdm import tqdm


def best_map(l1, l2):
    """
    Permute labels of l2 to match l1 as much as possible
    """
    if len(l1) != len(l2):
        print("L1.shape must == L2.shape")
        exit(0)

    label1 = np.unique(l1)
    n_class1 = len(label1)

    label2 = np.unique(l2)
    n_class2 = len(label2)

    n_class = max(n_class1, n_class2)
    g = np.zeros((n_class, n_class))

    for i in range(0, n_class1):
        for j in range(0, n_class2):
            ss = l1 == label1[i]
            tt = l2 == label2[j]
            g[i, j] = np.count_nonzero(ss & tt)

    aa = linear_assignment(-g)
    aa = np.asarray(aa)
    aa = np.transpose(aa)

    new_l2 = np.zeros(l2.shape)
    for i in range(0, n_class2):
        new_l2[l2 == label2[aa[i][1]]] = label1[aa[i][0]]
    return new_l2.astype(int)


def evaluation(x_selected, n_clusters, y):
    """
    This function calculates ARI, ACC and NMI of clustering results

    Input
    -----
    X_selected: {numpy array}, shape (n_samples, n_selected_features}
            input data on the selected features
    n_clusters: {int}
            number of clusters
    y: {numpy array}, shape (n_samples,)
            true labels

    Output
    ------
    nmi: {float}
        Normalized Mutual Information
    acc: {float}
        Accuracy

        k_means = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300,
                     tol=0.0001, precompute_distances=True, verbose=0,
                     random_state=None, copy_x=True, n_jobs=1)
    """

    k_means = KMeans(n_clusters=n_clusters, tol=0.0001)
    k_means.fit(x_selected)
    y_predict = k_means.labels_ + 1

    # calculate NMI
    nmi = normalized_mutual_info_score(y, y_predict)

    # calculate ACC
    y_permuted_predict = best_map(y, y_predict)
    acc = accuracy_score(y, y_permuted_predict)

    return nmi, acc


def dataset_pro(data_name, methods):
    """
    data sets preprocessing
    :param data_name: .m file, 'X': n,d; 'Y': n,1
    :return: 'X': n,d; 'Y': n,1
    """
    data_old = scio.loadmat('D:/Exp/Exp-Datasets/' + data_name + '.mat')
    label = data_old["Y"].astype('int')  # n 1
    unique_label = np.unique(label)
    classes = unique_label.shape[0]
    if methods == 'minmax':
        minmaxscaler = sklearn.preprocessing.MinMaxScaler()
        x = minmaxscaler.fit_transform(data_old["X"])
    elif methods == 'scale':
        x = sklearn.preprocessing.scale(data_old["X"])  # n d
    else:
        x = np.array(data_old["X"])
    return x, label.reshape((label.shape[0],)), classes


def dataset_info(data_name, methods):
    """
    calculate data information
    :param data_name: .m file, 'X': n,d; 'Y': n,1
    :return: 'X': n,d; 'Y': n,1
    """
    data_old = scio.loadmat('D:/Exp/Exp-Datasets/' + data_name + '.mat')
    label = data_old["Y"].astype('int')  # n 1
    label_count = pd.DataFrame(data_old['Y'])
    label_info = label_count.value_counts()
    unique_label = np.unique(label)
    classes = unique_label.shape[0]
    return label.reshape((label.shape[0],)), classes, label_info


def cluster_evaluation(data, label, classes, idx, cluster_times, feature_nums):
    """
    different feature num with fixed cluster times
    :param classes:
    :param label:
    :param data: n d
    :param idx: feature weight idx
    :param cluster_times: 20
    :param feature_nums: [50, 100, 150, 200, 250, 300]
    """
    nmi_fs_cluster_times = []
    acc_fs_cluster_times = []

    for feature_num in feature_nums:

        x_selected = data[:, idx[:feature_num]]
        nmi_cluster_times = []
        acc_cluster_times = []
        for i in range(cluster_times):
            nmi, acc = evaluation(x_selected, classes, label)
            nmi_cluster_times.append(nmi)
            acc_cluster_times.append(acc)
        # print('feature num:', feature_num)
        # print(" NMI: {}+/-{}".format(np.mean(nmi_cluster_times, 0), np.std(nmi_cluster_times, 0)))
        # print(" ACC: {}+/-{}".format(np.mean(acc_cluster_times, 0), np.std(acc_cluster_times, 0)))

        nmi_fs_cluster_times.append([np.mean(nmi_cluster_times, 0), np.std(nmi_cluster_times, 0)])
        acc_fs_cluster_times.append([np.mean(acc_cluster_times, 0), np.std(acc_cluster_times, 0)])
    # print('cluster evaluation done')
    return np.array(nmi_fs_cluster_times), np.array(acc_fs_cluster_times)


def write_excel(file_path, nmi, acc, data_name, para_best, feature_nums):
    """
    save the nmi and acc result
    :param feature_nums:
    :param para_best: save para
    :param acc:
    :param file_path:
    :param nmi: list; feature_nums: (1: mean; 2: std)
    :param data_name:
    """

    f = xlwt.Workbook()
    sheet1 = f.add_sheet(data_name, cell_overwrite_ok=True)
    # add table head
    sheet1.write(0, 0, data_name)
    sheet1.write(0, 1, 'mean')
    sheet1.write(0, 2, 'std')
    # write f_num in the r+1 row
    f_num = len(feature_nums)
    for r in range(f_num):
        sheet1.write(r + 1, 0, 'nmi-' + str(feature_nums[r]))
        # save mean and std
        sheet1.write(r + 1, 1, nmi[r][0])
        sheet1.write(r + 1, 2, nmi[r][1])

        sheet1.write(f_num + 1 + r + 1, 0, 'acc-' + str(feature_nums[r]))
        # save mean and std
        sheet1.write(len(nmi) + 1 + r + 1, 1, acc[r][0])
        sheet1.write(len(nmi) + 1 + r + 1, 2, acc[r][1])

    f.save(file_path + data_name + para_best + '.csv')  # save
    # print('save to excel--- done!')


def write_to_excel_one(file_path, data_name, nmi, acc, paras):
    """
    write one para result to excel
    :param file_path:
    :param data_name:
    :param nmi:
    :param acc:
    :param paras:[1,2,3]
    :return:
    """
    nmi_acc = np.vstack((nmi, acc))
    data = pd.DataFrame(nmi_acc)
    paras_str = [str(para) for para in paras]
    data.columns = paras_str
    writer = pd.ExcelWriter(file_path + data_name + '-paras.xlsx')
    data.to_excel(writer, 'NMI_ACC')
    writer.save()


def write_to_excel_w(file_path, data_name, w, paras):
    """
    write the weight matrix to excel
    :param file_path:
    :param data_name:
    :param w: weight matrix
    :param paras:
    :return:
    """
    data = pd.DataFrame(w)
    paras_str = [str(para) for para in paras]
    data.columns = paras_str
    writer = pd.ExcelWriter(file_path + data_name + '-w.xlsx')
    data.to_excel(writer, 'w')
    writer.save()


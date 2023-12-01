from sklearn.metrics.pairwise import pairwise_distances
from utility.local_learning_func import estimateReg, eig_lastk
from utility.unsupervised_evaluation import dataset_pro, cluster_evaluation
from tqdm import tqdm
import traceback
import LLSRFS
import numpy as np

eps = np.spacing(1)

if __name__ == '__main__':
    Cluster_times = 20
    Feature_nums = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for data_name in ['lung_discrete']:
        print("data name:", data_name)
        X, Y, Classes = dataset_pro(data_name, 'scale')
        n_samples, n_features = X.shape

        Dist_x = pairwise_distances(X) ** 2
        Local_reg, S = estimateReg(Dist_x, 15)
        S = (S + S.T) / 2
        Ls = np.diag(S.sum(0)) - S
        F, E_val = eig_lastk(Ls, Classes)

        Sl_all = []
        for ik in range(n_features):
            Sl_all.append(pairwise_distances(X[:, ik].reshape(n_samples, 1)) ** 2)
        Sl_all = np.array(Sl_all)
        np.save(data_name + '.npy', Sl_all)

        paras = [0.001, 0.01]#, 0.1, 1, 10, 100, 1000]
        paras2 = [0.01, 0.1]#, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        grid_search = [(lamb_b, lamb_p, lamb_r) for lamb_p in paras for lamb_r in paras for lamb_b in paras2]
        # result in diff para
        nmi_result = np.zeros((len(Feature_nums), len(grid_search)))
        acc_result = np.zeros((len(Feature_nums), len(grid_search)))

        kk = 0
        for Para in tqdm(grid_search):
            # result with temp para
            try:
                W, S = LLSRFS.LLSRFS(X.T, Classes, Para, S, F, Local_reg, Sl_all)
                Idx = np.argsort(W, 0)[::-1]

                # store K-means result with ave 20-times
                nmi_para_temp, acc_para_temp = cluster_evaluation(X, Y, Classes, Idx, Cluster_times, Feature_nums)

                # restore result of temp para(means only)
                nmi_result[:, kk] = nmi_para_temp[:, 0]
                acc_result[:, kk] = acc_para_temp[:, 0]

            except(Exception, BaseException) as e:
                print(e)
                print(repr(e))
                exstr = traceback.format_exc()
                print(exstr)
                print('error')
            kk += 1
        print("nmi:", np.max(nmi_result, 1))
        print("acc:", np.max(acc_result, 1))

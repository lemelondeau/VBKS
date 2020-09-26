"""
@Tong
04-02-2018
-----
Monte carlo sampling
"""
import distance
import KerConstruction as kc
# import multiprocessing as mp
from sobol_lib import *
from scipy.stats import norm
import hmc
import numpy as np


# sample from prior
# low-dependency sequence
def hyper_samples_qmc(means, sigmas, num_samples):
    # sample from N(0,1) first then make transformation
    s = np.empty([num_samples, len(means)])

    # Sampling with quasi Monte Carlo
    u = i4_sobol_generate(len(means), num_samples, 100)  # sobol

    # Inversion
    n_0 = norm.ppf(u)

    # Transformation
    for i in range(len(means)):
        n = n_0[i].T
        n = n * sigmas[i]
        n = n + means[i]
        s[:, i] = n.copy()
    return s


# sample from posterior
def hyper_samples_hmc(m, num_samples, Lmax, stepsize):
    """
    :param m: a TRAINED gp model
    :param num_samples: how many samples
    :param stepsize:
    :return:
    """
    optimizer, _, _ = hmc.sample_skleargp_hyper(m, num_samples, Lmax, stepsize)
    return optimizer


# a Bayesian way to get the distance, Monte Carlo
def kernel_distance_b(k1, k2, hypers1, hypers2, X):
    num_samples = hypers1.shape[0]
    ker_dis = []
    for i in range(num_samples):
        k1 = k1.clone_with_theta(hypers1[i, :])
        k2 = k2.clone_with_theta(hypers2[i, :])
        # k1.theta = hypers1[i, :] Can't use this method to change kernel hypers
        # k2.theta = hypers2[i, :]

        covmat1 = k1(X)
        covmat2 = k2(X)
        d = distance.distance(covmat1, covmat2)
        ker_dis.append(d)

    return np.mean(ker_dis)


# pairwise hypers (if there are n samples of hyper for each kernel, compute n times of distance)
def kernel_distance_b_mat(covmat1, covmat2):
    num_samples = covmat1.shape[0]
    ker_dis = []
    for i in range(num_samples):
        try:
            d = distance.distance(covmat1[i], covmat2[i])
            ker_dis.append(d)
        except Exception as e:
            print(e)
            print(i)
        else:
            pass
        finally:
            pass

    return np.mean(ker_dis)


# a Bayesian way to get the distance matrix
def get_dismat_b(kernels1, X, num_samples, hypers1_array, kernels2=None, hypers2_array=None):
    """
    :param kernels1: string-formed kernels, array
    :param X: nd array, input of training dataset
    :param kernels2: string-formed kernels
    :return: distance matrix between kernels1 and kernels2
    """
    if kernels2 is None:
        ker_num = kernels1.shape[0]
        dismat = np.zeros([ker_num, ker_num])

        # get the upper tri matrix
        for i in range(ker_num):
            ker_i = kc.str2ker(kernels1[i])  # string-formed kernel to real kernel
            hyper_i = hypers1_array[i]
            for j in range(i + 1, ker_num, 1):
                ker_j = kc.str2ker(kernels1[j])
                hyper_j = hypers1_array[j]
                # time0=time.time()
                dismat[i, j] = kernel_distance_b(ker_i, ker_j, hyper_i, hyper_j, X)
                # time1=time.time()
                # print(time1-time0)
        # form the whole matrix, diagonal is zero
        dismat = dismat + dismat.T

    else:
        ker1_num = kernels1.shape[0]
        ker2_num = kernels2.shape[0]
        dismat = np.empty([ker1_num, ker2_num])
        for i in range(ker1_num):
            ker_i = kc.str2ker(kernels1[i])
            hyper_i = hypers1_array[i]
            for j in range(ker2_num):
                ker_j = kc.str2ker(kernels2[j])
                hyper_j = hypers2_array[j]
                dismat[i, j] = kernel_distance_b(ker_i, ker_j, hyper_i, hyper_j, X)

    return dismat


# a Bayesian way to get the distance matrix
# pre-generate covmat, avoiding repeating
def get_dismat_b_mat(covmats1, covmats2=None):
    if covmats2 is None:
        ker_num = len(covmats1)
        dismat = np.zeros([ker_num, ker_num])

        # get the upper tri matrix
        for i in range(ker_num):
            covmat_i = covmats1[i]
            for j in range(i + 1, ker_num, 1):
                covmat_j = covmats1[j]
                dismat[i, j] = kernel_distance_b_mat(covmat_i, covmat_j)
        # form the whole matrix, diagonal is zero
        dismat = dismat + dismat.T

    else:
        ker1_num = len(covmats1)
        ker2_num = len(covmats2)
        dismat = np.empty([ker1_num, ker2_num])
        for i in range(ker1_num):
            covmat_i = covmats1[i]
            for j in range(ker2_num):
                covmat_j = covmats2[j]
                dismat[i, j] = kernel_distance_b_mat(covmat_i, covmat_j)

    return dismat

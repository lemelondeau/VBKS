"""
@Tong
04-02-2018
----
Implementation of BO for model selection:
Malkomes, G.; Schaff, C.; and Garnett, R. 2016. Bayesian optimization for automated model selection. In Proc. NeurIPS, 2900–2908.
Bayesian optimization implementation with EI
Distance as input
"""
import numpy as np
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import CustomRBF, WhiteKernel, ConstantKernel as C

import ModelEvidence as me
import KernelDistance as kd
import KerConstruction as kc
import distance
import time


class BO:
    def __init__(self, f, domain, data_x, data_y, iter, initnum, bayesian=None,
                 num_samples=10, posterior=None):
        """
        :param f: GP model, to compute the model evidence
        :param domain: initial kernel domain
        :param data_x: data for computing the distance
        :param data_y:
        :param iter: how many BO iterations
        :param initnum: how many initial kernels
        :param bayesian: Bayesian hypers or optimized hypers
        :param num_samples: number of hyper samples if Bayesian
        :param posterior: sample the hyper: prior distribution or posterior distribution
        """
        self.f = f
        self.data_x = data_x
        self.data_y = data_y
        self.iter = iter
        self.initnum = initnum
        self.mins = []
        self.bayesian = bayesian
        self.num_samples = num_samples
        self.posterior = posterior
        self.start_time = time.time()
        self.time_elapsed = []
        self.record_best_ker = []
        if self.bayesian is None:
            self.DXX, self.Y, self.DZZ, self.DXZ, self.observations, self.candidates, _, _ = self.bo_init(
                domain)
        else:
            self.DXX, self.Y, self.DZZ, self.DXZ, self.observations, self.candidates, self.obs_hypers, self.candi_hypers = self.bo_init(
                domain)
        self.time_elapsed.append(time.time() - self.start_time)

    # acquisition function
    def EI(self, mu, min, sigma, maxormin):
        if maxormin == 'min':
            z = (min - mu) / sigma
        else:
            z = (mu - min) / sigma
        Phi = norm.cdf(z)
        phi = norm.pdf(z)
        EI = sigma * (z * Phi + phi)
        return EI

    def one_iter(self):
        k = C(1.0) * CustomRBF(1.0) + WhiteKernel(1.0)
        gp = GaussianProcessRegressor(kernel=k, alpha=0.01, normalize_y=True, n_restarts_optimizer=10)
        gp.custom_fit(self.DXX, self.Y)
        mu, std = gp.custom_predict(self.DZZ, self.DXZ, return_std=True)
        # record the best kernel ever seen
        min_value = np.min(self.Y)
        best_ker = self.observations[np.argmin(self.Y)]
        self.record_best_ker.append(best_ker)
        # remember to reshape mu and std
        ei = self.EI(mu.reshape(-1), min_value, std.reshape(-1), 'min')
        next_i = np.argmax(ei)

        # update candidate
        if self.bayesian is None:
            self.update_candidates_nonbayesian(next_i)
        else:
            self.update_candidates_bayesian(next_i)

        return min_value

    def get_covmat(self, ker_string):
        cov_mat = []
        # if self.bayesian is None:
        for i in range(len(ker_string)):
            kernel = kc.str2ker(ker_string[i])
            myme = me.ModelEvidence(self.data_x, self.data_y, kernel)
            covmat = myme.m.kernel_(self.data_x)
            cov_mat.append(covmat)

        return cov_mat

    # TODO: optimize memory space
    # I encountered a severe problem changing double to single in matlab, would it be influenced here?
    def get_covmat_b(self, ker_string, hypers):
        covmats = []
        for i in range(len(ker_string)):
            ker = kc.str2ker(ker_string[i])
            hyper = hypers[i]
            covmat = np.empty(shape=(len(hyper), len(self.data_x), len(self.data_x)), dtype=np.float32)
            for j in range(len(hyper)):
                # ker.theta = hyper[j, :]
                ker = ker.clone_with_theta(hyper[j, :])
                covmat[j] = ker(self.data_x)
                covmat[j][np.diag_indices_from(covmat[j])] += 0.01  # self.alpha
            covmats.append(covmat)
        return covmats

    def get_hyper_sample(self, ker_string):
        hypers = []
        if self.posterior is None:
            for i in range(len(ker_string)):
                mu, sigma = kc.prior_params(ker_string[i])
                hyper = kd.hyper_samples_qmc(mu, sigma, self.num_samples)
                hypers.append(hyper)
        else:
            for i in range(len(ker_string)):
                kernel = kc.str2ker(ker_string[i])
                hyper_me = me.ModelEvidence(self.data_x, self.data_y, kernel, prior=None)
                hyper = kd.hyper_samples_hmc(hyper_me.m, self.num_samples, Lmax=30, stepsize=1e-2)
                hypers.append(hyper)
        return hypers

    def bo_init(self, domain):
        print("Initializing...")

        # discrete domain, contains the string-formed kernels
        initial_kernels = domain[0:self.initnum]
        candidate_kernels = domain[self.initnum:]
        initial_y = np.empty([self.initnum, 1])
        for i in range(self.initnum):
            initial_y[i] = self.f(initial_kernels[i])

        if self.bayesian is None:
            # obtain covariance matrices
            print("Generating covmat...")
            obs_mat = self.get_covmat(initial_kernels)
            candi_mat = self.get_covmat(candidate_kernels)

            print("Computing dismat...")
            initial_dxx = self.get_dismat_mat(obs_mat)
            initial_dzz = self.get_dismat_mat(candi_mat)
            initial_dxz = self.get_dismat_mat(obs_mat, kernels_covmat2=candi_mat)
            print("Initialization finished.")

            return initial_dxx, initial_y, initial_dzz, initial_dxz, initial_kernels, candidate_kernels, obs_mat, candi_mat
        else:
            print("Sampling hypers...")
            t0 = time.time()
            obs_hypers = self.get_hyper_sample(initial_kernels)
            candi_hypers = self.get_hyper_sample(candidate_kernels)
            t1 = time.time()
            total = t1 - t0
            print("Time elapsed: " + str(total))
            print("Sampling finished.")

            print("Generating covmat...")
            t0 = time.time()
            obs_mat = self.get_covmat_b(initial_kernels, obs_hypers)
            candi_mat = self.get_covmat_b(candidate_kernels, candi_hypers)

            t1 = time.time()
            total = t1 - t0
            print("Time elapsed: " + str(total))
            print("Generating finished.")

            print("Computing dismat...")
            t0 = time.time()
            initial_dxx = kd.get_dismat_b_mat(obs_mat)
            initial_dzz = kd.get_dismat_b_mat(candi_mat)
            initial_dxz = kd.get_dismat_b_mat(obs_mat, candi_mat)
            t1 = time.time()
            print("Time elapsed: " + str(t1 - t0))
            print("Computing finished.")
            print("Initialization finished.")

            return initial_dxx, initial_y, initial_dzz, initial_dxz, initial_kernels, candidate_kernels, obs_hypers, candi_hypers

    # run BO
    def bo_run(self):
        for i in range(self.iter):
            print("Iteration " + str(i))
            min_value = self.one_iter()
            self.mins.append(min_value)
            # attention: time_elapsed and mins don't match, shift 1 times because of initialization
            self.time_elapsed.append(time.time() - self.start_time)
        # results of the last iteration
        min_value = np.min(self.Y)
        self.mins.append(min_value)
        best_ker = self.observations[np.argmin(self.Y)]
        self.record_best_ker.append(best_ker)

    # update the candidate set and observations
    # update distance matrices
    # For optimized hyper
    def update_candidates_nonbayesian(self, next_i):

        def extend_dxx(index):
            # move one column from DXZ to DXX
            temp = self.DXZ[:, index]
            self.DXX = np.concatenate((self.DXX, temp[:, np.newaxis]), axis=1)
            self.DXX = np.concatenate((self.DXX, np.append(temp, 0)[np.newaxis, :]), axis=0)
            return

        def shrink_dzz(indices):
            self.DZZ = np.delete(self.DZZ, indices, 0)
            self.DZZ = np.delete(self.DZZ, indices, 1)
            return

        def extend_dxz_reuse(indices, dis_vector):
            dis_mat = np.delete(dis_vector, indices).reshape(1, -1)
            self.DXZ = np.concatenate((self.DXZ, dis_mat), axis=0)
            return


        def shrink_dxz(indices):
            self.DXZ = np.delete(self.DXZ, indices, 1)
            return

        # evaluate
        next_x = self.candidates[next_i]
        # next_x_covmat = [self.candi_mat[next_i]]  # list
        next_y = self.f(next_x)
        # add the evaluated one to observation set
        self.observations = np.append(self.observations, next_x)
        self.Y = np.append(self.Y, next_y)
        # self.obs_mat.append(self.candi_mat[next_i])
        extend_dxx(next_i)
        # remove the evaluated one from candidate set
        self.candidates = np.delete(self.candidates, next_i, 0)  # don't change the orders
        # del self.candi_mat[next_i]
        temp = self.DZZ[:, next_i]
        shrink_dzz(next_i)
        shrink_dxz(next_i)  # don't change the orders
        # extend_dxz(next_x_covmat)  # don't change the orders
        extend_dxz_reuse(next_i, temp)
        return

    # For sampled-hypers
    def update_candidates_bayesian(self, next_i):
        def extend_dxx(index):
            # move one column from DXZ to DXX
            temp = self.DXZ[:, index]
            self.DXX = np.concatenate((self.DXX, temp[:, np.newaxis]), axis=1)
            self.DXX = np.concatenate((self.DXX, np.append(temp, 0)[np.newaxis, :]), axis=0)
            return

        def shink_dzz(indices):
            self.DZZ = np.delete(self.DZZ, indices, 0)
            self.DZZ = np.delete(self.DZZ, indices, 1)
            return

        def extend_dxz_reuse(indices, dis_vector):
            dis_mat = np.delete(dis_vector, indices)
            self.DXZ = np.concatenate((self.DXZ, dis_mat), axis=0)
            return

        def shink_dxz(indices):
            self.DXZ = np.delete(self.DXZ, indices, 1)
            return

        # evaluate
        next_x = self.candidates[next_i]
        next_y = self.f(next_x)

        next_x_hyper = [self.candi_hypers[next_i]]  # list
        # add the evaluated one to observation set
        self.observations = np.append(self.observations, next_x)
        self.obs_hypers.append(next_x_hyper)
        self.Y = np.append(self.Y, next_y)
        extend_dxx(next_i)
        # remove the evaluated one from candidate set
        self.candidates = np.delete(self.candidates, next_i, 0)  # don't change the orders
        del self.candi_hypers[next_i]
        temp = self.DZZ[:, next_i]
        shink_dzz(next_i)
        shink_dxz(next_i)  # don't change the orders
        # extend_dxz(np.array([next_x]), next_x_hyper)  # don't change the orders
        extend_dxz_reuse(next_i, temp)
        return

    def get_dismat_mat(self, kernels_covmat1, kernels_covmat2=None):
        """
        :param kernels_covmat1: list of covariance matrices
        :param kernels_covmat2: list of covariance matrices
        :return: matrix
        """

        if kernels_covmat2 is None:

            ker_num = len(kernels_covmat1)
            dismat = np.zeros([ker_num, ker_num])

            # get the upper tri matrix
            for i in range(ker_num):
                covmat_i = kernels_covmat1[i]
                for j in range(i + 1, ker_num, 1):
                    covmat_j = kernels_covmat1[j]
                    dismat[i, j] = distance.distance(covmat_i, covmat_j)
            # form the whole matrix, diagonal is zero
            dismat = dismat + dismat.T

        else:
            ker1_num = len(kernels_covmat1)
            ker2_num = len(kernels_covmat2)
            dismat = np.empty([ker1_num, ker2_num])
            for i in range(ker1_num):
                covmat_i = kernels_covmat1[i]
                for j in range(ker2_num):
                    covmat_j = kernels_covmat2[j]
                    dismat[i, j] = distance.distance(covmat_i, covmat_j)

        return dismat

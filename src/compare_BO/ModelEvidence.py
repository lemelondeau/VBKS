"""
@Tong
21-01-2018
approximate Bayesian model evidence for a GP model using MCMC, BIC, Laplace Approximation
Use package sklearn
"""
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import numdifftools as nd
from sklearn.metrics import mean_squared_error
import hmc


class ModelEvidence:
    def __init__(self, X, y, k, prior=None):
        self.X = X
        self.y = y
        self.kernel = k
        self.prior = prior
        self.m, self.log_lik = self.train_gp()
        self.hyper_array = np.exp(self.m.kernel_.theta.copy())

    # train the GP model use a certain kernel, get the marginal likelihood
    # return the gp model for further usage
    def train_gp(self):
        m = GaussianProcessRegressor(kernel=self.kernel, alpha=0.01,
                                     normalize_y=True, n_restarts_optimizer=5)
        m.fit(self.X, self.y)
        return m, m.log_marginal_likelihood_value_

    def predict(self, m, x_test, y_test):
        predmu = m.predict(x_test)
        mse = mean_squared_error(y_test, predmu)
        return predmu, mse

    # an average prediction on hyper samples
    def hmc_predict(self, m, x_test, y_test, n_thin, samples):
        ms = []
        for s in samples[::n_thin]:
            m.kernel_ = m.kernel_.clone_with_theta(s)  # change the hypers in kernel
            mu = m.predict(x_test)
            ms.append(mu)
        avg_ms = np.mean(ms, 0)
        mse = mean_squared_error(y_test, avg_ms)
        return avg_ms, mse

    # get the second derivatives of neg log marginal likelihood
    def hessian(self, opt_hyper):
        def objective(hyperparams):
            theta = np.log(hyperparams)
            return -self.m.log_marginal_likelihood(theta)

        h = nd.Hessian(objective, method='forward')
        h_matrix = h(opt_hyper)
        return h_matrix

    # another numerical way to get Hessian, but doesn't seem to work
    def hessian_nu(self, opt_hyper):
        num_hypers = len(opt_hyper)
        h_matrix = np.zeros([num_hypers, num_hypers])

        delta = 1e-6
        f1 = self.gradient(opt_hyper)
        # _, f11 = self.m.log_marginal_likelihood(np.log(opt_hyper),eval_gradient=True)
        for i in range(num_hypers):
            dopt_hyper = opt_hyper.copy()
            dopt_hyper[i] = dopt_hyper[i] + delta
            f2 = self.gradient(dopt_hyper)
            h_matrix[:, i] = (f2 - f1) / delta

        h_matrix = (h_matrix + h_matrix.T) / 2
        return h_matrix

    # get the partial derivatives of neg log marginal likelihood
    # gradient at optimized hyper should be zero
    def gradient(self, opt_hyper):
        def objective(hyperparams):
            theta = np.log(hyperparams)
            return -self.m.log_marginal_likelihood(theta)

        g = nd.Gradient(objective, method='complex')
        gradients = g(opt_hyper)
        return gradients

    # get the second derivatives of neg log (marginal likelihood * prior)
    def hessian_post(self, opt_hyper):
        def objective(hyperparams):
            theta = np.log(hyperparams)
            return -self.m.log_marginal_likelihood(theta) - np.log(self.prior.pdf(opt_hyper))

        h = nd.Hessian(objective, method='forward')
        h_matrix = h(opt_hyper)
        return h_matrix

    # sampling from the hyperparameter posterior distribution
    # Use the optimized hypers as the initials
    def myhmc(self, num_samples, stepsize=5e-2):
        optimizer, params, neg_log_lik = hmc.sample_skleargp_hyper(self.m, num_samples, 30, stepsize)
        return params, optimizer, neg_log_lik

    def hmc_me(self):
        params, samples, neg_log_lik = self.myhmc(self.m, 100)
        # TODO: burn in needed
        return np.mean(neg_log_lik)

    # log_lik = log_lik_mode - M/2log_N
    def bic_me(self):
        me = -self.m.log_marginal_likelihood_value_
        me += 1 / 2 * self.m.kernel_.theta.shape[0] * np.log(len(self.y))
        return me

    # log_lik=log_lik_mode+log_prior_mode-1/2log_det_(-hessian)+d/2log_2pi
    # hessian() returns -hessian(log_lik)
    def laplace_me(self):
        me = - self.m.log_marginal_likelihood_value_

        if self.prior is None:
            h_matrix = self.hessian_nu(self.hyper_array.copy())
        else:
            h_matrix = self.hessian_post(self.hyper_array.copy())
            me -= np.log(self.prior.pdf(self.hyper_array.copy()))
        # TODO: det not positive (Hessian elements should be positive)
        s, logdet = np.linalg.slogdet(h_matrix)
        if s < 0:
            print("Hessian det negative!")
        me += 1 / 2 * logdet
        me -= 1 / 2 * self.m.kernel_.theta.shape[0] * np.log(2 * np.pi)
        return me

    def model_evidence(self, mode='hmc'):
        if mode == 'hmc':
            return self.hmc_me()
        elif mode == 'laplace':
            return self.laplace_me()
        else:
            return self.bic_me()

# if __name__ == "__main__":
#     filename = "../solar.csv"
#     data = pd.read_csv(filename, header=None)
#     data = np.array(data)
#     X = data[:, 0]
#     y = data[:, 1]
#     X = X.reshape(402, 1)
#     y = y.reshape(402, 1)
#     k1 = C(1.0) * RBF(length_scale=1) + WhiteKernel(1.0)
#     k2 = C(1.0) * RationalQuadratic(length_scale=1) + WhiteKernel(1.0)
#     k3 = DotProduct(1)
#     k = k1
#     # k.set_prior(GPy.priors.Gamma.from_EV(1.,10.)) # set the prior of hypers
#
#     # prior distribution of the hyper
#     num_param = k.theta.shape
#     from scipy.stats import multivariate_normal
#
#     prior = multivariate_normal(np.zeros(num_param), np.diag(np.ones(num_param)))
#     ME = ModelEvidence(X, y, k)

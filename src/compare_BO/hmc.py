"""
@Tong
25-01-2018
HMC implementation for GP hypers (sklearn)
sample_x0 is the optimized hyper
Used in ModelEvidence.py
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class HMC:
    def __init__(self, model=None):
        self.model = model

    def sample(self, f, sample_x0, num_samples, Lmax, epsilon):
        """

        :param f: likelihood fucntion, returns logprob, grad, param_if_x_is_not_whatwewant (logprob can't be float but can be np.float64)
        :param sample_x0: variables to sample
        :param num_samples: number of samples
        :param Lmax: leapfrog, num of steps
        :param epsilon: leapfrog, size of step
        :return: posterior samples of hyperparameter
        """
        # initialization
        D = sample_x0.size
        samples = np.zeros((num_samples, D))  # auxiliary optimizer
        samples_h = np.zeros((num_samples, D))  # real parameters
        neg_log_lik = np.zeros(
            num_samples)  # objective function, if there is a prior, it is marginal likelihood times prior
        samples[0] = sample_x0
        logprob, grad, samples_h[0] = f(sample_x0)
        neg_log_lik[0] = -logprob

        for t in range(num_samples - 1):
            # auxiliary p: random
            p = np.random.multivariate_normal(np.zeros(D), np.eye(D))
            x = samples[t].copy()
            p_t = p.copy()
            logprob_t = logprob.copy()  # not float
            grad_t = grad.copy()
            # Leapfrog
            """
            algotithm:
            1. Take a half step in time to update the momentum variable
            2. Take a full step in time to update the position variable
            3. Take the remaining half step in time to finish updating the momentum variable
            loggrad: log p(x)
            grad: -dU/dx=d log p(x)/dx
            
            p=p-epsilon/2*(dU/dx)
            """
            try:
                for s in range(Lmax):
                    # first half step
                    p = p + epsilon / 2 * grad
                    # full step
                    x += epsilon * p
                    # update logprob and grad
                    logprob, grad, params = f(x)
                    # last half step
                    p = p + epsilon / 2 * grad

                # whether to accept
                """
                U(x) = -log p(x) = - loggrad
                K(p) = 0.5 * p.dot(p)
                alpha= min(1, exp(-U(x')+U(x0)-K(p')+K(p0)))
                draw a random number u from Unif(0,1)
                u<=alpha, accept
                """
                alpha = np.exp(logprob - logprob_t - p.dot(p) / 2 + p_t.dot(p_t) / 2)
                if alpha > 1:
                    alpha = 1

                randomnum = np.random.uniform()
                if randomnum < alpha:
                    # accept
                    samples[t + 1] = x
                    samples_h[t + 1] = params
                    neg_log_lik[t + 1] = -logprob
                else:
                    # reject
                    samples[t + 1] = samples[t]
                    samples_h[t + 1] = samples_h[t]
                    neg_log_lik[t + 1] = neg_log_lik[t]
                    logprob, grad = logprob_t, grad_t
            except Exception as e:
                print(e)
                print(params)
                samples[t + 1] = samples[t]
                samples_h[t + 1] = samples_h[t]
                neg_log_lik[t + 1] = neg_log_lik[t]
                logprob, grad = logprob_t, grad_t
            else:
                pass
            finally:
                pass
        return samples, samples_h, neg_log_lik


def data_gene(datasize, test_size):
    X = np.linspace(0., datasize * 4 * np.pi, datasize * 100)[:, None]
    Y = -np.cos(X) + np.random.randn(*X.shape) * 0.3 + 1
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=test_size, random_state=None)
    return Xtrain, Xtest, Ytrain, Ytest


def sample_from_normal():
    def test_normal(x):
        mu = 1
        sigma = 1
        pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
        logprob = np.log(pdf)
        # grad = d(logprob)/dx =- dU/dx
        grad = -(x - mu) / sigma ** 2  # not the derivative of pdf
        return logprob, grad, x

    x0 = np.array([0])
    H = HMC()
    samples, _, _ = H.sample(test_normal, x0, 3000, 30, 0.01)
    samples = samples[1000:3000]
    print(np.mean(samples))
    print(np.std(samples))
    plt.figure(1)
    xmin = samples.min()
    xmax = samples.max()
    xs = np.linspace(xmin, xmax, 100)
    for i in range(samples.shape[1]):
        kernel = stats.gaussian_kde(samples[:, i])
        plt.subplot(211)
        plt.plot(xs, kernel(xs))
    plt.show()


def sample_gpy_hyper(m, num_samples, Lmax, epsilon):
    def f(x):
        # loggrad
        # self.model._transform_gradients(self.model.objective_function_gradients()): the gradient of the negative log_likelihood
        # self.model.objective_function(): -log(x): -float(self.log_likelihood()) - self.log_prior()

        # self.model._log_likelihood_gradients()
        # - self.model.objective_function_gradients()
        m.optimizer_array = x
        logprob = np.float64(- m.objective_function())
        grad = - m._transform_gradients(
            m.objective_function_gradients())  # don't know why need transform, but check with numdifftool
        hyper = m.unfixed_param_array.copy()
        return logprob, grad, hyper

    """
    Important:
    change m.optimizer_array in each iteration/ or in f(x), otherwise logprob and grab will never change
    """
    H = HMC()
    optimizers, params, neg_log_lik = H.sample(f, m.optimizer_array.copy(), num_samples, Lmax, epsilon)
    return optimizers, params, neg_log_lik


def sample_skleargp_hyper(m, num_samples, Lmax, epsilon):
    def f(x):
        logprob, grad = m.log_marginal_likelihood(x, eval_gradient=True)
        hyper = np.exp(x)
        # TODO: posterior
        return logprob, grad, hyper

    H = HMC()
    optimizers, params, neg_log_lik = H.sample(f, m.kernel_.theta.copy(), num_samples, Lmax, epsilon)
    return optimizers, params, neg_log_lik


def draw_posterior(samples):
    opt_hyper = samples[0, :]

    num_of_hypers = samples.shape[1]

    rows = np.ceil(num_of_hypers / 4)

    for i in range(num_of_hypers):
        xmin = samples[:, i].min()
        xmax = samples[:, i].max()
        xs = np.linspace(xmin, xmax, 100)
        # print(i)
        try:
            kde = stats.gaussian_kde(samples[:, i])
            plt.subplot(rows, 4, 1 + i)
            plt.plot(xs, kde(xs))
            plt.axvline(x=opt_hyper[i], ymin=0, ymax=1, color='r')
        except:
            pass
    plt.show()

# if __name__ == "__main__":
#     np.random.seed(1)
#     Xtrain, Xtest, Ytrain, Ytest = data_gene(2, 0.8)
#     kernel = GPy.kern.RBF(input_dim=1)
#     m = GPy.models.GPRegression(Xtrain, Ytrain, kernel.copy())
#     m.optimize()
#     np.random.seed(1)
#     samples, samples_h,_ = sample_gpy_hyper(m.copy(), 1000, 20, 0.01)
#     draw_posterior(samples_h)
#
#     np.random.seed(1)  # to make sure the random numbers in two hmc are the same
#     hmc = gpyhmc(m.copy(), stepsize=0.01)
#     samples, samples_h2,  _ = hmc.sample(num_samples=1000, hmc_iters=20)  #
#     draw_posterior(samples_h2)
#
#     np.random.seed(1)
#     gp = GaussianProcessRegressor(kernel=RBF(1.0), alpha=0.01,
#                                   normalize_y=True)
#     gp.fit(Xtrain, Ytrain)
#     samples, samples_h3, _ = sample_skleargp_hyper(gp, 1000, 20, 0.01)
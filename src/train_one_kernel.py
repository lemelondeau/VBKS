"""
@Tong
23-Jun-2019
Train one kernel with data.
delta distribution
"""
import numpy as np
from sklearn.metrics import mean_squared_error
from utils import training_utils as tu
import gpflow
from utils import ker_construction as kc
from math import sqrt
import time
import os
import pickle
import sys


class Logger(gpflow.actions.Action):
    def __init__(self, model, test_in, test_out):
        self.model = model
        self.logf = []
        self.start_time = time.time()
        self.time_elapsed = []
        self.test_in = test_in
        self.test_out = test_out
        self.log_rmse = []

    def run(self, ctx):
        # this is to save elbo every iteration, if save_logger==True
        if (ctx.iteration % 10) == 0:
            likelihood = - ctx.session.run(self.model.likelihood_tensor)
            self.logf.append(likelihood)
        # if (ctx.iteration % 10) == 0:
        #     y_pred = self.model.predict_y(self.test_in)
        #     self.log_rmse.append(sqrt(mean_squared_error(self.test_out, y_pred[0])))
        self.time_elapsed.append(time.time() - self.start_time)


# TODO: choose optimizer
def run_adam(model, iterations, X_test, y_test, savelog):
    opt = gpflow.train.AdamOptimizer().make_optimize_action(model)
    # opt = gpflow.train.AdagradOptimizer(0.01).make_optimize_action(model)
    logger = Logger(model, X_test, y_test)
    if savelog:
        actions = [opt, logger]
        gpflow.actions.Loop(actions, stop=iterations)()  # sgd training, save logger
    else:
        gpflow.actions.Loop(opt, stop=iterations)()  # sgd training
    model.anchor(model.enquire_session())
    return logger


class TrainOneKernel:
    def __init__(self, ker_str, X_train, y_train, X_test, y_test, minibatch_size,
                 init_hyper_fixed_first_run=False, Z_trainable=False, ker_bracket=None, save_logger=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.ker_str = ker_str
        self.minibatch_size = minibatch_size
        self.Z_trainable = Z_trainable  # whether to train the inducing input, true or flase
        self.init_hyper_fixed_first_run = init_hyper_fixed_first_run  # use pre-defined initial hypers in the first run
        self.save_trainables_folder = None  # if ture, save hypers and inducing variables
        self.reused_trainables = None  # previous saved hypers/trainables
        self.kern_init = None  # a initialized kernel (dict of hyper)
        self.ker_bracket = ker_bracket  # if True, operation * will multiply the current one with all the previous ones
        self.save_logger = save_logger  # if True, save log of elbo every few iterations

    def train_one_kernel_diff_subsets(self, num_iter, batch_sizes, global_u_size,
                                      local_u_size, global_u_itv=0, n_init=1, n_rerun=1, Z_fixed=False,
                                      sub_mode='random', interval=4):

        """
        :param num_iter: how many iters to train
        :param batch_sizes: different subset size to use
        :param global_u_size: size of global inducing points
        :param local_u_size: size of local inducing points
        :param global_u_itv: in case global_u_size is too big for a small subset, change global_u_size accordingly
        :param n_init: how many initial hypers for this kernel with a certain subset
        :param n_rerun: how many different subsets of same size to run (one subset contains many minibatches)
        :param Z_fixed: whether to use same Z for different subsets
        :param sub_mode: the way to select a subset
        :param interval: interval information for selecting the subset
        :return: training curves, prediction on validation datasets
        """
        train_results_diff_sizes = {'batch_sizes': batch_sizes, 'init_num': n_init, 'rerun_num': n_rerun}
        if Z_fixed:
            # self.X_train will be ordered
            Z = tu.get_inducing(self.X_train, global_u_size, local_u_size)
        for i in batch_sizes:
            subset_size = (i + 1) * self.minibatch_size
            elbo_loggers_diff_runs = []
            time_loggers_diff_runs = []
            elbos_diff_runs = []
            y_preds_diff_runs = []
            rmses_diff_runs = []
            time_elp_diff_runs = []
            pred_times_diff_runs = []
            for j in range(n_rerun):
                start_time = time.time()
                X_sub, y_sub = self.get_a_subset(subset_size, sub_mode, j, interval)
                # inducing points
                if not Z_fixed:
                    # uniformly distributed, when global_u_itv is not 0, global_u_size is changed accordingly
                    if global_u_itv == 0:
                        Z = tu.get_inducing(X_sub, global_u_size, local_u_size)
                    else:
                        global_u_size = np.int(np.round(subset_size / global_u_itv))
                        Z = tu.get_inducing(X_sub, global_u_size, local_u_size)
                elbo_loggers, time_loggers, elbos_estimate, y_preds, rmses, pred_times = self.train_one_kernel_diff_init(X_sub,
                                                                                                             y_sub, Z,
                                                                                                             num_iter,
                                                                                                             n_init)
                # append the results of same size subsets together
                # loggers_diff_runs[i][j]: ith run, jth initial hypers
                elbo_loggers_diff_runs.append(elbo_loggers)
                time_loggers_diff_runs.append(time_loggers)
                elbos_diff_runs.append(elbos_estimate)
                y_preds_diff_runs.append(y_preds)
                rmses_diff_runs.append(rmses)
                pred_times_diff_runs.append(pred_times)
                time_elp_diff_runs.append(time.time() - start_time)
            # the results of different runs same subset size are saved in a dict
            train_results_diff_runs = {'elbo_loggers': elbo_loggers_diff_runs,
                                       'time_loggers': time_loggers_diff_runs,
                                       'elbos': elbos_diff_runs,
                                       'y_preds': y_preds_diff_runs,
                                       'rmses': rmses_diff_runs,
                                       'pred_time_elp': pred_times_diff_runs,
                                       'time_elp': time_elp_diff_runs}
            # then save the above dict in to a big dict
            train_results_diff_sizes['batchnum' + str(i)] = train_results_diff_runs
        return train_results_diff_sizes

    def train_one_kernel_diff_init(self, X, y, Z, num_iter, n_init=1):
        """
        :param X: input
        :param y: output
        :param Z: inducing input
        :param num_iter: number of training iteration
        :param n_init: number of initial hypers
        :return:
        """
        # for training subsets, try-catch is outside
        elbo_loggers = []
        time_loggers = []
        elbos_estimate = []
        y_preds = []
        rmses = []
        pred_times = []
        for i in range(n_init):
            gpflow.reset_default_graph_and_session()
            # Construct a kernel with the kernel string
            # kern_init will be assigned later
            if i == 0:
                kern = kc.str2ker(self.ker_str, X.shape[1], self.init_hyper_fixed_first_run, self.ker_bracket)
            else:
                kern = kc.str2ker(self.ker_str, X.shape[1], False, self.ker_bracket)
            # training
            logger, y_pred, rmse, elbo, pred_time = self.train_once(kern, X, y, Z, num_iter)
            elbo_loggers.append(logger.logf)
            time_loggers.append(logger.time_elapsed)
            elbos_estimate.append(elbo)
            y_preds.append(y_pred)
            rmses.append(rmse)
            pred_times.append(pred_time)
        return elbo_loggers, time_loggers, elbos_estimate, y_preds, rmses, pred_times

    def train_once(self, kern, X, y, Z, num_iter):
        """
        the dict of hyper, likelihood and inducing variable can be saved
        and be reused when more data is included for training
        """
        # assign_kern_init: assign the initial hypers from outside
        assign_kern_init = self.kern_init
        if assign_kern_init is not None:
            kern.assign(assign_kern_init)
        # The number of inducing points has a big influence on the time of newing a SVGP object
        m = gpflow.models.SVGP(X, y, kern, gpflow.likelihoods.Gaussian(), Z=Z, minibatch_size=self.minibatch_size,
                               q_diag=False)
        m.feature.set_trainable(self.Z_trainable)
        # reuse previous hypers/trainables
        if self.reused_trainables is not None:
            m.assign(self.reused_trainables)
        logger = run_adam(m, num_iter, self.X_test, self.y_test, self.save_logger)
        # estimate the elbo
        instance_num = np.int(X.shape[0]/self.minibatch_size/2)
        instance_num = np.max([100, instance_num])
        evals = [m.compute_log_likelihood() for _ in range(instance_num)]
        elbo_estimated = np.mean(np.array(evals))
        # make prediction
        start_pred_time = time.time()
        y_pred, rmse = self.prediction(m)
        end_pred_time = time.time()
        pred_time = end_pred_time - start_pred_time
        # save the final hypers
        if self.save_trainables_folder is not None:
            hyper_dict = m.kern.read_values()
            lik_dict = m.likelihood.read_values()
            svgp_dict = m.read_values()
            # mkdir if not exist
            if not os.path.exists(self.save_trainables_folder):
                os.makedirs(self.save_trainables_folder)
            filename = self.save_trainables_folder + self.ker_str + '.pkl'
            with open(filename, "wb") as fout:
                pickle.dump(hyper_dict, fout)
                pickle.dump(lik_dict, fout)
                pickle.dump(svgp_dict, fout)
        return logger, y_pred, rmse, elbo_estimated, pred_time

    # --------------------------------------------------------------------------
    def train_one_kernel_full_data(self, global_u_size, local_u_size, num_iter, n_init=1, save_kern_init_folder=None):
        Z = tu.get_inducing(self.X_train, global_u_size, local_u_size)
        elbo_loggers = []
        time_loggers = []
        elbos_estimate = []
        rmse_loggers = []
        y_preds = []
        rmses = []
        pred_times = []
        # kernel hyperparameter is assigned through self.kern_init, if self.kern_init is not None
        for i in range(n_init):
            # try:
            gpflow.reset_default_graph_and_session()
            if i == 0:
                kern = kc.str2ker(self.ker_str, self.X_train.shape[1], self.init_hyper_fixed_first_run,
                                  self.ker_bracket)
            else:
                kern = kc.str2ker(self.ker_str, self.X_train.shape[1], False, self.ker_bracket)

            # save the initial kernel (dict for tensor) for replication
            if save_kern_init_folder is not None:
                # mkdir if not exist
                if not os.path.exists(save_kern_init_folder):
                    os.makedirs(save_kern_init_folder)
                filename = save_kern_init_folder + self.ker_str + '_' + str(i) + '.pkl'
                ker_dict = kern.read_values()
                with open(filename, "wb") as fout:
                    pickle.dump(ker_dict, fout)

            logger, y_pred, rmse, elbo, pred_time = self.train_once(kern, self.X_train, self.y_train, Z, num_iter)
            elbo_loggers.append(logger.logf)
            time_loggers.append(logger.time_elapsed)
            rmse_loggers.append(logger.log_rmse)
            elbos_estimate.append(elbo)
            y_preds.append(y_pred)
            rmses.append(rmse)
            pred_times.append(pred_time)
            # except Exception as e:
            #     print("The No." + str(i) + "initial hypers is not valid.")
            #     print('str(e):\t\t', str(e))
                # elbo_loggers.append(None)
                # time_loggers.append(None)
                # rmse_loggers.append(None)
                # elbos_estimate.append(None)
                # y_preds.append(None)
                # rmses.append(None)
                # pred_times.append(None)
        # results of different intial hypers are save in lists
        return elbo_loggers, time_loggers, elbos_estimate, y_preds, rmses, rmse_loggers, pred_times

    # --------------------------------------------------------------------------
    def prediction(self, m):
        y_pred = m.predict_y(self.X_test)
        rmse = sqrt(mean_squared_error(self.y_test, y_pred[0]))
        y_preds_mean = y_pred[0]
        y_preds_var = y_pred[1]
        mnlp = np.mean(0.5 * np.log((2 * np.pi) * y_preds_var) +
                       0.5 * (((y_preds_mean - self.y_test) ** 2) / y_preds_var))
        return y_pred, rmse

    def get_a_subset(self, subset_size, mode, randseed, interval):
        """
        get a subset to train with, use different ways to get the subset
        :param subset_size:
        :param mode: the way to select data points for the subset, "random", "consecutive", "uniform", "consecutive_itv"
        :param randseed: if mode="random", this is the random seed; if mode="consecutive", this is related to the starting point
        :param interval: "consecutive_itv" (choose the points every interval points)
        :return:
        """
        # TODO: selecting test data across the whole dataset while choosing training data consecutively is not good
        # how to choose test data? consecutive test data after training data/split X_sub into train and test
        if mode == "random":
            np.random.seed(randseed)
            indices = np.arange(self.X_train.shape[0])
            np.random.shuffle(indices)
            X_sub = self.X_train[indices[1:subset_size], :]
            y_sub = self.y_train[indices[1:subset_size], :]
        elif mode == "consecutive":
            start_batch = randseed
            X_sub = self.X_train[start_batch * subset_size:(start_batch + 1) * subset_size, :]
            y_sub = self.y_train[start_batch * subset_size:(start_batch + 1) * subset_size, :]
        elif mode == "consecutive_itv":
            start_batch = randseed
            start_point = start_batch * subset_size
            end_point = start_point + (subset_size - 1) * interval + 1
            X_sub = self.X_train[start_point:end_point:interval, :]
            y_sub = self.y_test[start_point:end_point:interval, :]
        elif mode == "uniform":
            interval = np.round(self.X_train.shape[0] / subset_size)
            X_sub = self.X_train[::interval, :]
            y_sub = self.y_train[::interval, :]
        else:
            print('No such mode for selecting subset points!')
            sys.exit(1)

        return X_sub, y_sub

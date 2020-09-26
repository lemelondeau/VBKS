"""
@Tong
23-Jun-2019

"""
from utils import training_utils as tu
from sklearn.model_selection import train_test_split
import train_one_kernel as to
import numpy as np
import pickle
import os
import time
import traceback
import utils.get_init_dict as gidc


# training multiple kernels and return results
def train_full_data_multi_ker(datafile, data_to_use_size, global_u_size, local_u_size, multi_ker_str, iter_num,
                              minibatch_size,
                              scale,
                              test_size, init_hyper_fixed_first_run=False, multi_ker_init_dict=None, n_init=1,
                              save_folder=None,
                              ker_name_file='trained_ker.txt', ker_bracket=None, save_logger=True):
    # save_folder: if None, the results will not be saved
    X_all, y_all = tu.load_data(datafile)
    X_to_use = X_all[1:data_to_use_size, :]
    y_to_use = y_all[1:data_to_use_size, :]
    X_to_use, y_to_use = tu.data_processing(X_to_use, y_to_use, scale=scale)
    X_train, X_test, y_train, y_test = train_test_split(X_to_use, y_to_use, test_size=test_size, random_state=2019)

    #   TODO: validation and test data, they are different

    if save_folder is not None:
        fname = save_folder + "testdata" + ".pkl"
        if not os.path.exists(fname):
            test_data = {'X_test': X_test, 'y_test': y_test}
            with open(fname, "wb") as fout:
                pickle.dump(test_data, fout)

    train_results_multi_ker = {}
    valid_ker = []
    ELBOs = []
    for ker_str in multi_ker_str:
        try:
            st = time.time()

            T = to.TrainOneKernel(ker_str, X_train, y_train, X_test, y_test, minibatch_size,
                                  init_hyper_fixed_first_run=init_hyper_fixed_first_run, Z_trainable=False,
                                  ker_bracket=ker_bracket, save_logger=save_logger)
            if multi_ker_init_dict is not None:
                T.kern_init = multi_ker_init_dict[ker_str]
            # TODO What if set Z_trainable to be true (cannot reuse parameters)
            # training, n_init times
            elbo_loggers_diff_init, time_loggers_diff_init, elbos_estimate_diff_init, \
            y_preds_diff_init, rmses_diff_init, rmse_loggers_diff_init, \
            pred_time_elp = \
                T.train_one_kernel_full_data(global_u_size, local_u_size,
                                             iter_num, n_init=n_init, save_kern_init_folder=save_folder + 'init_kern/')
            total_time_per_run = (time.time() - st) / n_init
            # loggers: including time and elbo and rmse for different initial hypers (a list)
            train_results_one_ker = {'elbo_loggers': elbo_loggers_diff_init,
                                     'time_loggers': time_loggers_diff_init,
                                     'elbos': elbos_estimate_diff_init,
                                     'y_preds': y_preds_diff_init,
                                     'rmses': rmses_diff_init,
                                     'rmse_logger': rmse_loggers_diff_init,
                                     'pred_time_elp': pred_time_elp,
                                     'total_t_per_run': total_time_per_run}
            train_results_multi_ker[ker_str] = train_results_one_ker
            valid_ker.append(ker_str)
            ELBOs.append(elbos_estimate_diff_init)

            # save training results when one kernel is trained
            if save_folder is not None:
                fname = save_folder + ker_str + ".pkl"
                with open(fname, "wb") as fout:
                    pickle.dump(train_results_one_ker, fout)
                # also record the kernel string
                with open(ker_name_file, 'a') as f:
                    f.write(ker_str)
                    f.write('\n')
        except Exception as e:
            print('Something is wrong with' + ker_str)
            print('str(e):\t\t', str(e))
        finally:
            print(ker_str + " is finished")
    # train_results_multi_ker[ker_str]['loggers'][i].logf, the results of ith initial
    return valid_ker, ELBOs, train_results_multi_ker, X_test, y_test


# train multiple kernels with subsets of a certain size, multiple runs,  multiple initial hypers, save results
# reuse hypers from previous subset's training results as initial hypers
#     This method can get the training results of multiple kernels, the trained hypers and inducing variables are saved
#     by setting T.save_trainables_folder.
#     The training results of each kernel is saved in save_folder_child.
#     T.reused_trainables is the hyper/trainables from previous training. If this is None, the the hypers in kernel are
#     random or use the assigned values. (multi_ker_trainables_pre is None)
def train_subset_multi_ker_reuse_hyper(datafile, data_to_use_size, global_u_size, local_u_size, multi_ker_str, iter_num,
                                       minibatch_size,
                                       scale,
                                       test_size, subset_batchnum, save_folder_parent,
                                       n_rerun=1, n_init=1,
                                       ker_trainables_pre_settings=None,
                                       Z_fixed=False, multi_ker_init_dict=None,
                                       ker_name_file='trained_ker.txt', sub_mode='random',
                                       ker_bracket=None, random_state=2019, save_logger=True):
    """
    :param datafile: dataset to use D
    :param data_to_use_size: choose the first data_to_use_size to use as D_use
    :param global_u_size:
    :param local_u_size:
    :param multi_ker_str: multiple kernels to train (can also be just one kernel)
    :param iter_num: training iteration
    :param minibatch_size:
    :param scale: range of X
    :param test_size:
    :param subset_batchnum: scalar, select subset_batchnum*minibatch_size point from D_use for training
    :param n_rerun: number of subsets for the same size
    :param n_init: how many set of initial hypers
    :param ker_trainables_pre_settings: path and settings of the trainables from previous training results,
    None means do not use previous results.
    If this is not None, reused_trainables is not None, n_init should be set to 1
    :param save_folder_parent: a folder to save all the results for different size of subsets
    :param Z_fixed: whether to use the same Z for different subsets
    :param multi_ker_init_dict: assign values for hypers
    :param ker_name_file: save the name of valid kernels
    :return:
    """
    # subset_batchnum contains one subset size only in my VBKS experiments, this is because I want all the kernels
    # be trained for this certain subset_batchnum before adding more data.
    subset_batchnum = np.array([subset_batchnum])
    X_all, y_all = tu.load_data(datafile)
    X_to_use = X_all[1:data_to_use_size, :]
    y_to_use = y_all[1:data_to_use_size, :]
    X_to_use, y_to_use = tu.data_processing(X_to_use, y_to_use, scale=scale)
    # Multiple runs: use different training data
    # For multiple runs, the test data is two-folded, one part is the same for all runs, another part is different
    # according to the random seed. (their size is slightly different)
    X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X_to_use, y_to_use, test_size=test_size / 2,
                                                                random_state=2019)

    X_train, X_test, y_train, y_test = train_test_split(X_train_0, y_train_0, test_size=test_size / 2,
                                                        random_state=random_state)

    X_test = np.concatenate((X_test, X_test_0), axis=0)
    y_test = np.concatenate((y_test, y_test_0), axis=0)

    if save_folder_parent is not None:
        fname = save_folder_parent + "testdata" + ".pkl"
        if not os.path.exists(fname):
            test_data = {'X_test': X_test, 'y_test': y_test}
            with open(fname, "wb") as fout:
                pickle.dump(test_data, fout)
    train_results_multi_ker = {}
    ELBOs = []
    valid_ker = []
    save_folder_child = save_folder_parent + 'train_res_' + str(subset_batchnum[0]) + '/'
    # remember to mkdir once the path is defined!
    if not os.path.exists(save_folder_child):
        os.mkdir(save_folder_child)
    # TODO probably remove this
    all_ker_name_file = save_folder_child + 'trained_ker.txt'
    if not os.path.exists(all_ker_name_file):
        with open(all_ker_name_file, 'w') as f:
            f.write('valid_ker\n')
    with open(save_folder_child + ker_name_file, 'w') as f:
        f.write('valid_ker\n')
    for ker_str in multi_ker_str:
        print('Training ' + ker_str + ':')
        try:
            st = time.time()
            if ker_trainables_pre_settings is None:
                # do not use trained hyper from previous subset as initials
                reused_trainables = None
                # it is usually the first subset, assign the initial kernel hypers
                if multi_ker_init_dict is not None:
                    print('Use assigned initial hypers.')
                    best_init_kern = multi_ker_init_dict[ker_str]
                else:
                    best_init_kern = None
            else:
                reused_trainables = gidc.get_saved_trainables(ker_str,
                                                              ker_trainables_pre_settings['pre_working_folder'],
                                                              ker_trainables_pre_settings['Z_reused'],
                                                              ker_trainables_pre_settings['lik_reused'])

                # if reused_trainables is not None:
                #     print('reuse previous trained hypers')
                # else:
                #     print('Something is wrong with' + ker_str)
                #     print('Cannot fetch trainables!')

                best_init_kern = None

            T = to.TrainOneKernel(ker_str, X_train, y_train, X_test, y_test, minibatch_size,
                                  init_hyper_fixed_first_run=False, Z_trainable=False,
                                  ker_bracket=ker_bracket, save_logger=save_logger)
            # whether to reuse hypers/trainables
            T.kern_init = best_init_kern
            T.reused_trainables = reused_trainables
            T.save_trainables_folder = save_folder_parent + 'trainables_batchnum' + str(subset_batchnum[0]) + '/'
            # train_results_one_ker is a dict
            """
            train_results_diff_runs = {'elbo_loggers': elbo_loggers_diff_runs,
                                       'time_loggers': time_loggers_diff_runs,
                                       'elbos': elbos_diff_runs,
                                       'y_preds': y_preds_diff_runs,
                                       'rmses': rmses_diff_runs,
                                       'time_elp': time_elp_diff_runs}
            """
            train_results_one_ker = T.train_one_kernel_diff_subsets(iter_num, subset_batchnum, global_u_size,
                                                                    local_u_size,
                                                                    n_init=n_init, n_rerun=n_rerun,
                                                                    Z_fixed=Z_fixed,
                                                                    sub_mode=sub_mode)

            avg_time_per_run = (time.time() - st) / n_init / n_rerun
            train_results_one_ker['batchnum' + str(subset_batchnum[0])]['avg_t_per_run'] = avg_time_per_run

            # save the above dict into a big dict with kernel name as key value
            train_results_multi_ker[ker_str] = train_results_one_ker
            ELBOs_curr_ker = train_results_one_ker['batchnum' + str(subset_batchnum[0])]['elbos']
            ELBOs.append(ELBOs_curr_ker)
            valid_ker.append(ker_str)

            # save the training results and ELBOs
            fname = save_folder_child + ker_str + ".pkl"
            with open(fname, "wb") as fout:
                pickle.dump(ELBOs_curr_ker, fout)
                pickle.dump(train_results_one_ker, fout)
            # remove previous trainables
            if ker_trainables_pre_settings is not None:
                gidc.remove_saved_trainables(ker_str, ker_trainables_pre_settings['pre_working_folder'])
            # also record the kernel string
            # TODO probably remove this
            with open(save_folder_child + ker_name_file, 'a') as f:
                f.write(ker_str)
                f.write('\n')
            with open(all_ker_name_file, 'a') as f:
                f.write(ker_str)
                f.write('\n')
        except KeyError as e:
            print('KeyError. No initials or no trainables.')
            print(str(e))
        except Exception as e:
            print('Something is wrong with' + ker_str)
            # traceback.print_exc()
            print('str(e):\t\t', str(e))
        finally:
            print(ker_str + " is finished")
    # record when all kernels is finished (this is to check whether training is done when multiple gpus are used)
    # each subset_batchnum have one record_file
    # TODO probably remove this
    record_file = save_folder_parent + 'record' + str(subset_batchnum[0]) + '.txt'
    with open(record_file, 'a') as f:
        f.write(ker_name_file)
        f.write('\n')
    # ELBOs[i][j][k], ith kernel, jth run, kth initial hyper
    # train_results_multi_ker[ker_str][batchnum20]['elbos'][j][k]
    return valid_ker, ELBOs, train_results_multi_ker, X_test, y_test

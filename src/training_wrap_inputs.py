"""
wrap all settings to make it terse when calling the function (compare to train_multi_kernels)
as well as easy to read settings from configuration files
"""
import os
import train_multiple_kernels as tm
import utils.get_init_dict as gidc


def train_one_subset(datafile, settings, subset_batchnum, working_folder, multi_ker_str,
                     multi_ker_init_dict, reuse_batchnum,
                     ker_name_file):
    """
    Train multiple kernels with one subset of data.
    :param datafile:
    :param settings: training setting
    :param subset_batchnum: scalar
    :param working_folder: results path
    :param multi_ker_str: kernels
    :param multi_ker_init_dict: kernel hypers initial value
    :param reuse_batchnum: scalar
    :param ker_name_file: log trained kernels
    :return:
    """
    # set multi_ker_init_dict=None if do not specify initial hypers
    # if don't want to reuse parameters, set reuse_batch_sizes=0
    # first get the settings
    data_to_use = settings['data_to_use']
    global_u_size = settings['global_u_size']
    local_u_size = settings['local_u_size']
    iter_num = settings['iter_num_subset']
    minibatch_size = settings['minibatch_size']
    scale = settings['scale']
    test_size = settings['test_size']
    n_init = settings['n_init']
    Z_fixed = settings['Z_fixed']
    Z_reused = settings['Z_reused']
    lik_reused = settings['lik_reused']
    random_state = settings['random_s']
    sub_mode = settings['sub_mode']
    ker_bracket = settings['ker_bracket']
    save_logger = settings['save_logger']

    if not os.path.exists(working_folder):
        os.mkdir(working_folder)

    # the hypers (trainables) to use as initial values is indicated by reuse_batch_sizes,
    # if 0 then do not reuse hypers,
    if reuse_batchnum == 0:
        # train with subsets
        valid_ker, elbos, train_results_multi_ker, X_test, y_test \
            = tm.train_subset_multi_ker_reuse_hyper(datafile,
                                                    data_to_use,
                                                    global_u_size,
                                                    local_u_size,
                                                    multi_ker_str,
                                                    iter_num,
                                                    minibatch_size,
                                                    scale, test_size,
                                                    subset_batchnum,
                                                    n_init=n_init,
                                                    Z_fixed=Z_fixed,
                                                    save_folder_parent=working_folder,
                                                    multi_ker_init_dict=multi_ker_init_dict,
                                                    ker_name_file=ker_name_file,
                                                    sub_mode=sub_mode,
                                                    ker_bracket=ker_bracket,
                                                    random_state=random_state,
                                                    save_logger=save_logger)
    else:
        pre_working_folder = working_folder + 'trainables_batchnum' + str(reuse_batchnum) + '/'
        k_t_pre_settings = {'pre_working_folder': pre_working_folder, 'Z_reused': Z_reused, 'lik_reused': lik_reused}
        # train with subsets, multi_ker_trainables_pre is not None
        valid_ker, elbos, train_results_multi_ker, X_test, y_test \
            = tm.train_subset_multi_ker_reuse_hyper(datafile,
                                                    data_to_use,
                                                    global_u_size,
                                                    local_u_size,
                                                    multi_ker_str,
                                                    iter_num,
                                                    minibatch_size,
                                                    scale, test_size, subset_batchnum,
                                                    n_init=n_init,
                                                    ker_trainables_pre_settings=k_t_pre_settings,
                                                    save_folder_parent=working_folder,
                                                    Z_fixed=Z_fixed,
                                                    ker_name_file=ker_name_file,
                                                    sub_mode=sub_mode,
                                                    ker_bracket=ker_bracket,
                                                    random_state=random_state)
    return valid_ker, elbos, train_results_multi_ker, X_test, y_test


def train_fulldata(datafile, settings, multi_ker_str, multi_ker_init_dict, working_folder="iter_pk_test/"):
    """
    Train multiple kernels with fulldata.
    :param datafile:
    :param settings:
    :param multi_ker_str:
    :param multi_ker_init_dict: assign initial hypers, if not None set n_init=1
    :param working_folder:
    :return:
    """
    # multi_ker_init_dict is None means do not used specified kernel initial hypers
    # first get the settings
    data_to_use = settings['data_to_use']
    global_u_size = settings['global_u_size']
    local_u_size = settings['local_u_size']
    iter_num = settings['iter_num_full']
    minibatch_size = settings['minibatch_size']
    scale = settings['scale']
    test_size = settings['test_size']
    n_init = settings['n_init']
    save_logger = settings['save_logger']
    ker_bracket = settings['ker_bracket']

    if multi_ker_init_dict is not None:
        n_init = 1

    if not os.path.exists(working_folder):
        os.mkdir(working_folder)

    valid_ker, elbos, train_results_multi_ker, X_test, y_test \
        = tm.train_full_data_multi_ker(datafile,
                                       data_to_use,
                                       global_u_size,
                                       local_u_size,
                                       multi_ker_str,
                                       iter_num,
                                       minibatch_size,
                                       scale, test_size,
                                       n_init=n_init,
                                       save_folder=working_folder,
                                       ker_bracket=ker_bracket,
                                       save_logger=save_logger,
                                       multi_ker_init_dict=multi_ker_init_dict,
                                       init_hyper_fixed_first_run=False)
    # train_results_multi_ker[ker_str]['elbo_loggers'][i], the results of ith initial

    return valid_ker, elbos, train_results_multi_ker, X_test, y_test

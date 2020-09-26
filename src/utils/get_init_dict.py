import pickle
import os
import sys


# todo invalid initial causing non positive definite and no/empty result saved?
# as such, assign default initialization
# tell if None
def get_best_init(multi_ker_str, working_folder):
    if not os.path.exists(working_folder):
        print('No such folder: ' + working_folder)
        sys.exit(1)
    multi_ker_init_dict = {}
    for ker_str in multi_ker_str:
        # which one has the largest elbo
        ker_train_results = working_folder + ker_str + ".pkl"
        if os.path.exists(ker_train_results):
            with open(ker_train_results, "rb") as fin:
                train_res = pickle.load(fin)
                elbos_per_ker = train_res["elbos"]
                i = elbos_per_ker.index(max(elbos_per_ker))
                kern_filename = working_folder + "init_kern/" + ker_str + "_" + str(i) + ".pkl"
                with open(kern_filename, 'rb') as fin2:
                    best_ker_dict = pickle.load(fin2)
                    multi_ker_init_dict[ker_str] = best_ker_dict

    return multi_ker_init_dict


def get_specified_init(multi_ker_str, working_folder, run_id):
    if not os.path.exists(working_folder):
        print('Kernel initials path does not exist!')
        sys.exit(1)
    multi_ker_init_dict = {}
    for ker_str in multi_ker_str:
        # which one has the largest elbo
        ker_train_results = working_folder + ker_str + ".pkl"
        if os.path.exists(ker_train_results):
            kern_filename = working_folder + "init_kern/" + ker_str + "_" + str(run_id) + ".pkl"
            with open(kern_filename, 'rb') as fin2:
                best_ker_dict = pickle.load(fin2)
                multi_ker_init_dict[ker_str] = best_ker_dict

    return multi_ker_init_dict


def get_saved_trainables(ker_str, pre_working_folder, Z_reused, lik_reused):
    kern_filename = pre_working_folder + ker_str + ".pkl"
    with open(kern_filename, 'rb') as fin:
        hyper_dict = pickle.load(fin)
        lik_dict = pickle.load(fin)
        svgp_dict = pickle.load(fin)
        if Z_reused:
            ker_trainables = svgp_dict
        else:
            if lik_reused:
                ker_trainables = {**hyper_dict, **lik_dict}
            else:
                ker_trainables = hyper_dict
    return ker_trainables


def remove_saved_trainables(ker_str, pre_working_folder):
    kern_filename = pre_working_folder + ker_str + ".pkl"
    os.remove(kern_filename)
    return


def get_saved_init(multi_ker_str, working_folder):
    if not os.path.exists(working_folder):
        print('Kernel initials path does not exist!')
        sys.exit(1)
    multi_ker_init_dict = {}
    for ker_str in multi_ker_str:
        kern_filename = working_folder + ker_str + ".pkl"
        with open(kern_filename, 'rb') as fin:
            best_ker_dict = pickle.load(fin)
            multi_ker_init_dict[ker_str] = best_ker_dict
    return multi_ker_init_dict

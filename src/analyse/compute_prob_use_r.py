"""
@Tong
04-07-2019
compute p(k) for a bunch of kernels
"""
import pickle
import numpy as np
import pandas as pd
import os
import paths

# os.environ['R_HOME'] = '/home/tengtong/miniconda3/envs/py36/lib/R'
os.environ['R_HOME'] = paths.r_home
from rpy2.robjects import r


def compute_prob(size_sample, prior_pg, r_read_file_name="20k_test_elbos.csv",
                 prob_result_file="20k_test_prob_result.csv", working_folder="./"):
    # save args in a txt file for R script yo read
    # todo don't run compute_prob in parallel... filename.txt will be changed
    # names_file = prob_result_file[:-4] + '.txt'
    # save_path = 'filenames_for_r/'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    text_file = open('filename.txt', "w")
    text_file.write(r_read_file_name)
    text_file.write('\n')
    text_file.write(prob_result_file)
    text_file.write('\n')
    text_file.write(working_folder)
    text_file.write('\n')
    text_file.write(str(size_sample))
    text_file.write('\n')
    text_file.write(str(prior_pg))
    text_file.write('\n')
    text_file.close()
    # kernel and local ELBOs
    # training_results = "20k_test_elbos.pkl",
    # with open(training_results, "rb") as fin:
    #     valid_ker = pickle.load(fin)
    #     elbos = pickle.load(fin)

    # save as csv file
    # "kernels" "L_i"
    # data = {'kernels': valid_ker,
    #         'L_i': np.array(elbos).reshape(-1)}
    # print(data)
    # df = pd.DataFrame(data)
    # print(df)
    # df.to_csv(r_read_file_name, index=None)
    # compute probability
    r.setwd('~/BKS/src/R_bks')
    r.source('bks_run_global_python.R')

# compute_prob(1000,1)

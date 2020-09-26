"""
If don't want to reuse parameters, set reuse_batch_sizes=0
Train a group of kernels using subsets of data on a certain GPU. subset size specified by batch_sizes
args:
1 gpu_id: which GPU to use
2 random_state: random seeds for sampling subsets
3 kernel group number, 4 start position, 5 group size: which kernels to train
6 cfg_file: configuration file
"""
import os
import pandas as pd
import numpy as np
from utils.use_tensorflow import use_gpu
import sys
import utils.get_init_dict as gidc
import utils.parse_cfg as cfg
import training_wrap_inputs as twi

gpu_id = int(sys.argv[1])
use_gpu(gpu_id)
group = int(sys.argv[2])
random_seed = int(sys.argv[3])  # multiple runs

start = int(sys.argv[4])
group_size = int(sys.argv[5])
ker = pd.read_csv('kernelstring3.csv')
multi_ker_str = np.array(ker).reshape(-1)[start + group * group_size:start + (group + 1) * group_size]

cfg_file = '../config/' + sys.argv[6]
settings, paths, working_folder = cfg.get_settings(cfg_file, 'sub')

num_batch = int(settings['num_of_batch_per_subset'])
num_subset = int(settings['num_of_subset'])
batch_sizes = np.array(range(num_subset)) * num_batch + num_batch
reuse_batch_sizes = np.array(range(num_subset)) * num_batch

datafile = paths['datafile']
multi_ker_init_dict = None
if paths['init_hyper_file'] is not None:
    multi_ker_init_dict = gidc.get_saved_init(multi_ker_str, paths['init_hyper_file'])

settings['random_s'] = random_seed
settings['initfile'] = paths["init_hyper_file"]
print(settings)

working_folder = '../results/' + working_folder + '_rnd' + str(random_seed) + '/'
if not os.path.exists(working_folder):
    os.mkdir(working_folder)
df = pd.DataFrame(settings, index=[0])
df.to_csv(working_folder + "settings.csv")

ker_name_file = "kernel_group" + str(group) + ".txt"
for i in range(batch_sizes.shape[0]):
    s = batch_sizes[i]
    twi.train_one_subset(datafile, settings, s, working_folder, multi_ker_str, multi_ker_init_dict,
                         reuse_batch_sizes[i], ker_name_file)

"""
Train a group of kernels using fulldata on a certain GPU.
args:
1 gpu_id: which GPU to use
2 kernel group number, 3 start position, 4 group size: which kernels to train
5 cfg_file: configuration file
"""
import os
import pandas as pd
import numpy as np
import training_wrap_inputs as twi
from utils.use_tensorflow import use_gpu
import sys
import utils.parse_cfg as cfg
import utils.get_init_dict as gidc


gpu_id = int(sys.argv[1])
use_gpu(gpu_id)
group = int(sys.argv[2])

start = int(sys.argv[3])
group_size = int(sys.argv[4])
ker = pd.read_csv('kernelstring3.csv')
multi_ker_str = np.array(ker).reshape(-1)[start + group * group_size:start + (group + 1) * group_size]

cfg_file = '../config/' + sys.argv[5]
settings, paths, working_folder = cfg.get_settings(cfg_file, "full")

datafile = paths['datafile']
multi_ker_init_dict = None
if paths['init_hyper_file'] is not None:
    multi_ker_init_dict = gidc.get_saved_init(multi_ker_str, paths['init_hyper_file'])

working_folder = '../results/' + working_folder + "/"
if not os.path.exists(working_folder):
    os.mkdir(working_folder)
settings['initfile'] = paths["init_hyper_file"]
print(settings)
df = pd.DataFrame(settings, index=[0])
df.to_csv(working_folder + "setting.csv")
twi.train_fulldata(datafile, settings, multi_ker_str, multi_ker_init_dict, working_folder)

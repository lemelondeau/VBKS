# Scalable Variational Bayesian Kernel Selection for Sparse Gaussian Process Regression
This repo contains the source code for the paper
[Scalable Variational Bayesian Kernel Selection for Sparse Gaussian Process Regression](https://arxiv.org/abs/1912.02641)
by Tong Teng, Jie Chen, Yehong Zhang and Bryan Kian Hsiang Low,
appearing in AAAI 2020.

## Setup

#### Requirements
Install R

python==3.6

requirements.txt (gpflow==1.5.1, tensorflow-gpu==1.14.0)
use ```conda install -c anaconda tensorflow-gpu==1.14``` to install for cuda10.1

Replace files in 
xxx/site-packages/sklearn/gaussian_process/gpr.py, kernels.py
 with 
src/dependencies/gpr.py, kernels.py
#### Install R packages
This will be use when computing the kernel posterior belief.

Related files are in src/bksgpR/
1. in env/env_maplecgpu.R, set R library directory and other directories
2. run src/bksgpR/install_packages.R (must use StanHeaders 2.17.2)
3. src/analyse/compute_prob_use_r.py: r.setwd('xxx/BKS/src/R_bks')

#### Paths Setting
src/paths.py



## How to run:
#### Files required
dataset file: e.g., 'datasets/swissgrid.csv'

configuration file: e.g., 'config/swiss_cfg_1.json'

kernel name file: e.g., 'src/kernelstring3.csv'

initial hyper file: e.g., 'config/swiss_init_14/'

##### Configuration
    "settings": {
    "data_to_use": number of data points to use in the experiment
    "global_u_size": number of global inducing points
    "local_u_size": number of local inducing points (in case of periodic pattern)
    "iter_num_full": how many iterations for full data training
    "iter_num_subset": how many iterations for subset training (no need of convergence)
    "minibatch_size": sgd minibatch size
    "scale": for time series data, if you don't want X to be in [0, 1]
    "test_size": test data proportion
    "n_init": how many different initial hypers for a kernel, if init_hyper_file is not None, this will be set to 1
    "ker_bracket": whether there are bracket in the kernel string
    "save_logger": default true, save the ELBO and time logger
    "Z_fixed": for subset training, default true, fix inducing points for different subsets
    "Z_reused": for subset training, default true, reuse SVGP inducing parameters from last subset training
    "lik_reused": for subset training, default true, reuse SVGP likelihood parameters from last subset training
    "sub_mode" : for subset training, "random", how to choose the subset
    "num_of_batch_per_subset": for subset training, how many bathces in a subset
    "num_of_subset": for subset training, how many subsets to train in the whole experiment
      },
      "paths": {
        "data_identifier": a string to identify data
        "datafile": e.g., "../datasets/swissgrid.csv",
        "init_hyper_file" : e.g., "../config/swiss_init/", see the following Hyperparameters section
        "working_folder_suffix": a string to add to the working folder name, it can be empty ''.
      }




#### Hyperparameters
1. If 'init_hyper_file' is  None: initial hyper will be generated randomly while creating the kernel.

2. To generate the init_hyper_file: run src/gen_init_hypers.py


#### Training
0. modify configuration files and *.sh files, mkdir 'out/' in 'src/'
1. exps_run_big.sh will call run_big_data.py, training on full data with multiple initial hypers or specified initial hypers

2. exps_run_subsets_reuse_true.sh will call run_subsets_reuse_true.py, set random_seeds to get the results for multiple runs.

**directory**: results will be save in result/working_folder (src/utils/parse_cfg.py generates working_folder name)

#### Show results
src/analyse/analyse_results.py: 

plot_rmse(): it will call kern_set_prob.py, BMA.py, plot_rmse.py in order. Changing of rmse with the increasing od data size is plotted.

compute_pk_large(): Changing of p(k) with the increasing od data size is plotted.

Run from src/analyse/: python analyse_results.py
Remember to set the right configuration file.

**directory**: working_folder/ana_res/, working_folder/plots_res/



####Compare with BO
1. Kernel selection with BO, run src/compare_BO/KernelSelectionBO.py. Remember to set the right configuration file.
2. Compare with VBKS, make sure plot_rmse() is done. Run src/compare_BO/plot_comparison.py. The plotted figure will be saved in full data working folder.


### Description of the scripts
#### Training
train_one_kernel.py -> train_multiple_kernels.py -> training_wrap_input.py->(run_big_data.py, run_subsets_reuse_true.py, 
run_small_kern_set_prob.py)

1. run_big_data.py: train with full data for multiple kernels
2. run_subsets_reuse_true.py: increase subset size, reuse hypers, save results for different kernels separately
reuse_batch_sizes = np.zeros() means do not reuse parameters (but the result is not good)
3. run_small_kern_set_prob.py: a small kernel set, increase subset size, save all results for different kernels in one file,
 compute probability p(k)
#### Analysis
1. kern_set_prob.py, gather trained results from run_subsets_reuse_true, compute p(k),
        calls compute_prob_use_r.py
2. BMA.py, get rmse
3. plot_rmse.py, plot and record rmse of best kernel and BMA rmse, and time
the results will be used to compare with BO
4. iter_prob.py, use all the data, compute p(k) in the middle of training
5. show_pk.py, plot p(k)

### Attention!

!!! Don't run compute_prob_use_r.py in parallel... filename.txt will be changed.




#!/usr/bin/env bash
groups=5 #number of gpu to use
gpus=(0 1 2 3 4)  #specify gpu, the total number = $groups
cfg_file='swiss_cfg_test.json'  #in folder config/
start_pos=0  # kernel id to start with, default 0
group_size=29  # how many kernels are trained on a single GPU
for ((i=0;i<$groups;i++))
do
    gpu=${gpus[$i]}
    #i: group number
    echo nohup python run_big_data.py ${gpu} ${i} ${start_pos} ${group_size} ${cfg_file}&
    nohup python run_big_data.py ${gpu} ${i} ${start_pos} ${group_size} ${cfg_file} > out/out${i}.txt &
done



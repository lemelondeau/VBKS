#!/usr/bin/env bash
seeds=( $1 )
gpu=$2
group=$3

start_pos=$4
group_size=$5
cfg_file=$6
for s in ${seeds[@]}
do
    echo nohup python run_subsets_reuse_true.py seed${s}_gpu${gpu}_group${group}
    nohup python run_subsets_reuse_true.py ${gpu} ${group}  ${s} ${start_pos} ${group_size} ${cfg_file} \
     > out/seed${s}_gpu${gpu}.txt 2>out/seed${s}_gpu${gpu}_err.txt
    wait
done
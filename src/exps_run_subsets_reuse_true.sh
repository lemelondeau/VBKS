#!/usr/bin/env bash
#gpus=(0 1 2 3 4)
groups=5 #number of gpu to use
gpus=(0 1 2 3 4)  #specify gpu, the total number = $groups
#seeds="1 2 3"
seeds="10"  # random seeds, this is an array
cfg_file='swiss_cfg_test.json'  #in folder config/
start_pos=0  # kernel id to start with
group_size=29  # how many kernels are trained on a single GPU

for ((i=0;i<$groups;i++))
do
    gpu=${gpus[$i]}
    echo $gpu
    group=$i
    echo $group
    ./exps_multiple_runs.sh "$seeds" $gpu $group ${start_pos} ${group_size} ${cfg_file} >> out/out.txt &
done

# ========
#for ((i=0;i<$groups;i++))
#do
#
#    gpu=$(($i%5))
#    echo nohup python run_subsets_reuse_true_dbg.py ${gpu} ${i}&
#    !!!! random_state
#    nohup python run_subsets_reuse_true_dbg.py ${gpu} ${i} > out/out${i}.txt &
#done

#for gpu in "${gpus[@]}"
#do
#    echo $gpu
#    group=$gpu
#    ./exps_multiple_runs.sh "$seeds" $gpu $group ${start_pos} ${group_size} ${cfg_file} >> out/out.txt &
#done
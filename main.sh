#!/bin/bash

# pretrain: bash main.sh pretrain
# downstream: bash main.sh downstream

main_task=$1

input_path='/home/swryu/downstream'
save_dir='/home/swryu/gearnet_checkpoint'
gpu="0 1 2 3"
enc_model="GearNet-Edge"
task_idx=0 # 0 ~ 4
use_ddp=True

load=False # If you want to load pre-trained weights


echo "Main task is $main_task"

if [ $main_task == "pretrain" ] ; then

        python -u main.py \
                --input_path $input_path \
                --save_dir $saver_dir \
                --use_ddp $use_ddp \
                --device_ids $gpu \
                --pt_encoder_type $enc_model \
                --pt_task $task_idx


elif [ $main_task == 'downstream' ] ; then

        python -u main.py \
                --input_path $input_path \
                --save_dir $save_dir \
                --use_ddp $use_ddp \
                --device_ids $gpu \
                --ft_encoder_type $enc_model \
                --ft_task $task_idx \
                --load_from_pretrained $load


else
        echo "Wrong options."

fi
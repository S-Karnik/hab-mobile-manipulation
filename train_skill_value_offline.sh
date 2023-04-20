#!/bin/bash

#SBATCH -o train_%j.out
#SBATCH -e train_%j.err
#SBATCH --cpus-per-task 20
#SBATCH --exclusive
#SBATCH --gres=gpu:volta:1
#SBATCH --ntasks 1
#SBATCH -N 1

source ~/.bashrc
conda activate hab-mm;
LD_LIBRARY_PATH=/state/partition1/llgrid/pkg/anaconda/anaconda3-2021a/lib/ python mobile_manipulation/train_skill_value_offline.py --cfg configs/rearrange/composite/set_table/mr.yaml --save-video all

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

LD_LIBRARY_PATH=/state/partition1/llgrid/pkg/anaconda/anaconda3-2021a/lib/ python mobile_manipulation/run_ppo_goal_residual.py --cfg configs/rearrange/skills/set_table_bilinear/open_fridge_v0A_joint_SCR.yaml --run-type train

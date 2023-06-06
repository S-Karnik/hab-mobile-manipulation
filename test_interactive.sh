#!/bin/bash

while true; do
    if nvidia-smi >/dev/null 2>&1; then
        echo "You are logged into a GPU node"
    else
        echo "You are not logged into a GPU node. Submitting job to run script"
        LLsub -i -s 20 -g volta:1 -T 2:00:00 &
        sleep 10
        node=$(LLstat | grep interactive | awk '{print $NF}')
        ssh -tt skarnik@$node << EOF
        cd ~/MEng/meng-project/hab-mobile-manipulation
LD_LIBRARY_PATH=/state/partition1/llgrid/pkg/anaconda/anaconda3-2021a/lib/ python mobile_manipulation/run_ppo_goal_residual.py --cfg configs/rearrange/skills/tidy_house_bilinear/place_v1_joint_SCR.yaml --run-type train
EOF
    fi
done

#!/bin/bash 
# Goal: have the direct path to the files, because there is a mismatch between where we can save things to in MTP tasks

# $2 == 031225_progressive_sf
if [[ "$1" == "task" ]]; then
    echo "MODE: task"
    export JOHN_MODE_FOR_TRAINING="task"
    export TASK_RUN_REPO="/opt/huawei/schedule-train/algorithm/arvd_repos/$2/"
    export DATASET_PATH="/opt/huawei/dataset/"
    export CHECKPOINT_SAVE="/opt/huawei/checkpoint/"
    export qwen3_vl_files="/opt/huawei/dataset/john_env/only_selfforcing_task/" # [][] CHECK IF THIS WORKS???? needs the task version of the flash_attn
    # export WHEELHOUSE_PATH="/opt/huawei/schedule-train/algorithm/wheelhouse"\
    export TORCH_HOME="/opt/huawei/dataset/self_forcing/lpips_models/torch/" # for lpips
    
    export VBENCH_CACHE_DIR=$TORCH_HOME 
    export VBENCH2_CACHE_DIR=$TORCH_HOME

elif [[ "$1" == "188" ]]; then
    echo "MODE: web (on 188)"
    export JOHN_MODE_FOR_TRAINING="webstudio"
    export TASK_RUN_REPO="/home/john_zhou/research/arvd_repos/"
    export DATASET_PATH="/shared/john/model_training/"
    export CHECKPOINT_SAVE="/shared/john/model_training/"
    # export qwen3_vl_files="/opt/huawei/dataset/john_env/only_selfforcing_task/" # [][] CHECK IF THIS WORKS???? needs the task version of the flash_attn
    # export WHEELHOUSE_PATH="/opt/huawei/schedule-train/algorithm/wheelhouse"\
    # export TORCH_HOME="/opt/huawei/dataset/self_forcing/lpips_models/torch/" # for lpips
    
    # export VBENCH_CACHE_DIR=$TORCH_HOME 
    # export VBENCH2_CACHE_DIR=$TORCH_HOME

else
    echo "mode: normal usage (aka webstudio)"
    export JOHN_MODE_FOR_TRAINING="webstudio"
    export TASK_RUN_REPO="/home/ma-user/work/algorithm/turbodiffusion/arvd_repos/$2/"
    export DATASET_PATH="/home/ma-user/work/dataset/"
    export CHECKPOINT_SAVE="/home/ma-user/work/algorithm/arvd_repos/"
    export qwen3_vl_files="/home/ma-user/work/dataset/john_env/only_selfforcing/"
    # export WHEELHOUSE_PATH="/home/ma-user/work/algorithm/arvd_repos/wheelhouse/"
    export TORCH_HOME="/home/ma-user/work/dataset/self_forcing/lpips_models/torch/" # for lpips
    
    export VBENCH_CACHE_DIR=$TORCH_HOME 
    export VBENCH2_CACHE_DIR=$TORCH_HOME
fi

echo "JOHN_MODE_FOR_TRAINING = ${JOHN_MODE_FOR_TRAINING}"
echo "TASK_RUN_REPO = ${TASK_RUN_REPO}"
echo "DATASET_PATH= ${DATASET_PATH}"
echo "Done with get_paths.sh"

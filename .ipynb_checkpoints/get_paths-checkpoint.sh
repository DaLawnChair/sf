#!/bin/bash 
# Goal: have the direct path to the files, because there is a mismatch between where we can save things to in MTP tasks

# $2 == 031225_progressive_sf
if [[ "$1" == "task" ]]; then
    echo "MODE: task"
    export JOHN_MODE_FOR_TRAINING="task"
    export TASK_RUN_REPO="/opt/huawei/schedule-train/algorithm/$2/"
    export DATASET_PATH="/opt/huawei/dataset/"
    export CHECKPOINT_SAVE="/opt/huawei/checkpoint/"
    export qwen3_vl_files="/opt/huawei/dataset/john_env/only_selfforcing_task/" # needs the task version of the flash_attn
    # export WHEELHOUSE_PATH="/opt/huawei/schedule-train/algorithm/wheelhouse"
else
    echo "MODE: normal usage (aka webstudio)"
    export JOHN_MODE_FOR_TRAINING="webstudio"
    export TASK_RUN_REPO="/home/ma-user/work/algorithm/arvd_repos/$2/"
    export DATASET_PATH="/home/ma-user/work/dataset/"
    export CHECKPOINT_SAVE="/home/ma-user/work/algorithm/arvd_repos/"
    export qwen3_vl_files="/home/ma-user/work/algorithm/arvd_repos/only_selfforcing/"
    # export WHEELHOUSE_PATH="/home/ma-user/work/algorithm/arvd_repos/wheelhouse/"
fi

echo "JOHN_MODE_FOR_TRAINING = ${JOHN_MODE_FOR_TRAINING}"
echo "TASK_RUN_REPO = ${TASK_RUN_REPO}"
echo "qwen3_vl_files = ${qwen3_vl_files}"
echo "Done with get_paths.sh"

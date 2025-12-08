#!/bin/bash 
# Goal: have the direct path to the files, because there is a mismatch between where we can save things to in MTP tasks


if [[ "$1" == "task" ]]; then
    echo "MODE: task"
    export JOHN_MODE_FOR_TRAINING="task"
    export TASK_RUN_REPO="/opt/huawei/schedule-train/algorithm/031225_progressive_sf/"
    export DATASET_PATH="/home/ma-user/work/dataset/"
    export CHECKPOINT_SAVE="/home/ma-user/work/algorithm/arvd_repos/"
    export qwen3_vl_files="/opt/huawei/dataset/john_env/only_selfforcing/"
    # export WHEELHOUSE_PATH="/opt/huawei/schedule-train/algorithm/wheelhouse"
else
    echo "MODE: normal usage (aka webstudio)"
    export JOHN_MODE_FOR_TRAINING="webstudio"
    export TASK_RUN_REPO="/home/ma-user/work/algorithm/arvd_repos/031225_progressive_sf/"
    export DATASET_PATH="/home/ma-user/work/dataset/"
    export CHECKPOINT_SAVE="/home/ma-user/work/algorithm/arvd_repos/"
    export qwen3_vl_files="/home/ma-user/work/algorithm/arvd_repos/only_selfforcing/"
    # export WHEELHOUSE_PATH="/home/ma-user/work/algorithm/arvd_repos/wheelhouse/"
fi

echo $JOHN_MODE_FOR_TRAINING
echo $qwen3_vl_files

echo "Done with get_paths.sh"

#!/bin/bash
set -e


run_task_folder="sf"

#================================= Boiler plate for environment set up
export NCLL_NVLS_ENABLE="0"

if  [[ -z "${JOHN_MODE_FOR_TRAINING}" ]]; then 
    echo "Not source, so perform sourcing and make environment"
    echo "INITIAL SET UP"
    cd "algorithm/arvd_repos/"$run_task_folder # pwd is /opt/huawei/schedule-train/
    echo "where am i?"
    pwd
    source ./get_paths.sh task $run_task_folder
    cd $TASK_RUN_REPO
    echo "echo $qwen3_vl_files"
    source ./make_env.sh
    
else
    echo "Already sourced (aka within ${JOHN_MODE_FOR_TRAINING}), so do not do any sourcing"
fi


## ================================== Actual changes
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
nproc_per_node=8
NNODES=1


instance_folder_name="260430_recreate_sf_standard_with_init"
config_name="260430_recreate_sf_fake_score/260430_recreate_sf_standard_with_init.yaml"

test_path="$DATASET_PATH"john_env/self_forcing_clone/self_forcing/model_training
# make log directory and save a version of the config
mkdir -p "$test_path"/$instance_folder_name
cp configs/$config_name "$test_path"/$instance_folder_name


echo "where am i?"
pwd
## ================================== Actual changes

echo "\n\n\n"
echo "\n\n\n"
echo "================================== ALL SYSTEM INFOS =================================="
nvcc --version
nvidia-smi
pip show torch
cat /etc/os-release
echo "================================== DONE LISTING ALL SYSTEM INFOS =================================="
echo "\n\n\n"
echo "\n\n\n"

# cd flash-attention
# python setup.py install


# ====================================================================
# This initial set up was taken from Samuel's run for FastWan distillation for LoRAs.
# Basic Info
# export WANDB_MODE="online"

export WANDB_MODE="disabled"
export WANDB_DISABLED="true"

export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export MASTER_PORT=29500
export TOKENIZERS_PARALLELISM=false
# export WANDB_BASE_URL="https://api.wandb.ai"
# export WANDB_MODE=online
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

# =========================
# Multi-node settings  ### NEW
# =========================

# How many nodes in the job (ModelArts gives you both of these)
NNODES=${MA_NUM_HOSTS:-${VC_WORKER_NUM:-1}}

# This node's rank (0,1,...,NNODES-1)
NODE_RANK=${VC_TASK_INDEX:-0}

# MASTER_ADDR = first hostname in VC_WORKER_HOSTS, or localhost fallback
if [[ -n "$VC_WORKER_HOSTS" ]]; then
  IFS=',' read -r MASTER_ADDR _ <<< "$VC_WORKER_HOSTS"
else
  MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
fi
export MASTER_ADDR

echo "[DIST] NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# End of Samuel's config file copy-over
# ====================================================================

torchrun --nnodes=$NNODES --nproc_per_node=$nproc_per_node --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR":"$MASTER_PORT \
  train.py \
  --config_path configs/$config_name \
  --logdir "$test_path"/$instance_folder_name/logs \
  --wandb-save-dir "$test_path"/$instance_folder_name


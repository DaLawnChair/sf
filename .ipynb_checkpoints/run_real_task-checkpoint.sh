#!/bin/bash

# INITAL SET UP
echo "INITIAL SET UP"

source ./get_paths.sh task
cd $TASK_RUN_REPO
echo "echo $qwen3_vl_files"
# echo $qwen3_vl_files

source ./make_env.sh


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
nproc_per_node=2
NNODES=1

torchrun --nnodes=$NNODES --nproc_per_node=$nproc_per_node --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR":"$MASTER_PORT \
  train.py \
  --config_path configs/self_forcing_dmd_recreate.yaml \
  --logdir logs/dmd_training \
  --disable-wandb




# Previous attempt: doesn't work
# LOCAL_RANK="0" 
# RANK="0" 
# WORLD_SIZE="1" 
# MASTER_ADDR="notebook-f352f131-5916-4bc0-b2a4-595c51570b41" 

# #python3 train.py \
# #        --config_path configs/self_forcing_ode.yaml \
# #        --logdir logs/ode_training


# torchrun --nnodes=$num_total_nodes --nproc_per_node=$num_gpus --rdzv_id=5235 \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint $MASTER_ADDR":"$MASTER_PORT \
#   train.py \
#   --config_path configs/self_forcing_ode.yaml \
#   --logdir logs/ode_training \
#   --disalbe-wandb



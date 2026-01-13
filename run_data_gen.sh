#!/bin/bash
echo "where am i?"
pwd

export NCLL_NVLS_ENABLE="0"
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" # OOM error suggests that this might work. Not sure what the ramifications are
# export PYTORCH_NO_CUDA_MEMORY_CACHING="1" # This removes caching and reduces reserved memory usage. Should make training slower though

run_task_folder="151225_naiive_progressive_sf"

if  [[ -z "${JOHN_MODE_FOR_TRAINING}" ]]; then 
    echo "Not source, so perform sourcing and make environment"
    echo "INITIAL SET UP"
    cd "algorithm/"$run_task_folder # pwd is /opt/huawei/schedule-train/
    echo "where am i?"
    pwd
    source ./get_paths.sh task $run_task_folder
    cd $TASK_RUN_REPO
    echo "echo $qwen3_vl_files"
    source ./make_env.sh
else
    echo "Already sourced (aka within ${JOHN_MODE_FOR_TRAINING}), so do not do any sourcing"
fi


python --version

python  - <<'PY'
import sys
 
try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True

except ModuleNotFoundError:
    print("ERROR: could not load flash_attn. Exit with error code 1.")
    FLASH_ATTN_AVAILABLE = False
    sys.exit(1)
PY


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

if  [[ "${JOHN_MODE_FOR_TRAINING}" == "task" ]]; then 
    if [[ $(nvidia-smi -L | wc -l) == 8 ]]; then
        export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" # this might be a problem for multinode
        nproc_per_node=8
    elif [[ $(nvidia-smi -L | wc -l) == 16 ]]; then
        export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" # this might be a problem for multinode
        nproc_per_node=16
    else
        echo "Error, neither 8 or 16 gpus"
        exit 1
    fi
    
    # export CUDA_VISIBLE_DEVICES="0,1"
    # nproc_per_node=2
else
    export CUDA_VISIBLE_DEVICES="0,1"
    nproc_per_node=2
fi

# instance_folder_name="essentially_bidirectional"
# config_name="essentially_bidirectional.yaml"

CUDA_VISIBLE_DEVICES="0,1" 
python scripts/create_lmdb_shards_video_latents.py \
    --data_path "$DATASET_PATH"self_forcing/self_forcing/bidirectional_diffusion_inference_videos_guidance5.0_wan1.3B_vidprom \
    --lmdb_path "$DATASET_PATH"self_forcing/self_forcing/latent_form_bidirectional_diffusion_inference_videos_guidance5.0_wan1.3B_vidprom_video_lmdb_2k/

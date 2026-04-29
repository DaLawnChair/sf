#!/bin/bash
export NCLL_NVLS_ENABLE="0"

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
# python setup.py develop
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
nproc_per_node=8
NNODES=1

# for making videos of latent form
torchrun --nnodes=$NNODES --nproc_per_node=$nproc_per_node --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR":"$MASTER_PORT \
  scripts/make_videos_latents_actually_latents.py \
  --config_path "${TASK_RUN_REPO}configs/default_config_bidirectional_diffusion.yaml" \
  --data_path "${TASK_RUN_REPO}prompts/vidprom_filtered_extended.txt" \
  --output_folder "${DATASET_PATH}john_env/self_forcing_clone/self_forcing/latent_form_bidirectional_diffusion_inference_videos_guidance6.0_wan1.3B_vidprom/" \
  --use_ema \
  --use_bidirectional
  
## For making the videos of video form
# torchrun --nnodes=$NNODES --nproc_per_node=$nproc_per_node --rdzv_id=5235 \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint $MASTER_ADDR":"$MASTER_PORT \
#   scripts/make_videos_latents.py \
#   --config_path "${TASK_RUN_REPO}configs/default_config_bidirectional_diffusion.yaml" \
#   --data_path "${TASK_RUN_REPO}prompts/vidprom_filtered_extended.txt" \
#   --output_folder "${DATASET_PATH}self_forcing/self_forcing/bidirectional_diffusion_inference_videos_guidance5.0_wan1.3B_vidprom/" \
#   --use_ema \
#   --use_bidirectional
#!/bin/bash
# NOTE
# this is an old copy that I had ran ages ago and hadn't had to change at all since opening up a Webstudio instance. `make_env.sh` is going to be the one to use going forward (until I get a newer one)

# set -e # if a command fails, do not perform any later commands


echo "091225 version"
pwd
############################  Fixing cuda  ############################
echo "NVIDIA_VISIBLE_DEVICES, $NVIDIA_VISIBLE_DEVICES"
export CUDA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"

echo "++++++++++"
echo $JOHN_MODE_FOR_TRAINING
echo "++++++++++"

if [[ $JOHN_MODE_FOR_TRAINING == "webstudio" ]]; then
    export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.183.01:/usr/lib/x86_64-linux-gnu/libcuda.so.535.183.01"
else # for task
    export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.161.08:/usr/lib/x86_64-linux-gnu/libcuda.so.535.161.08"
fi

# Adding cuda12.8
# export qwen3_vl_files="/home/ma-user/work/dataset/qwen3_vl_files"
export CUDA_HOME="${qwen3_vl_files%/}/cuda-12.8"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"

echo "WHERE IS qwen3_vl_files?"
echo $qwen3_vl_files

nvcc --version
nvidia-smi

############################  Fixing cuda  ############################


bash ../Miniconda3-py312_25.9.1-1-Linux-x86_64.sh -b -p "$HOME/miniconda3"

# unset PYTHONPATH
# echo $PYTHONPATH # this will be equivalent to the algorithm dir (initial pwd when loading the script)

current_path=$(pwd) # store the current path because we get reject back
$HOME/miniconda3/bin/conda init
source $HOME/.bashrc

export PATH="$(
  printf '%s' "$PATH" \
  | tr ':' '\n' \
  | awk -v p="$HOME/anaconda3" 'index($0,p)!=1' \
  | paste -sd: -
)"
export PATH="$HOME/miniconda3/bin:$PATH"

#conda activate
command -v python || type -a python
# command -v python3 || type -a python3
command -v pip || type -a pip


# # john: extra stuff from here on out
# export PATH="$HOME/miniconda3/bin:$PATH"

# # download .whl files from wheelhouse
# # pip install ${qwen3_vl_files%/}/wheelhouse/*.whl





# pip install --no-index --find-links=${qwen3_vl_files%/}/wheelhouse/ -r ${qwen3_vl_files%/}/webstudio_requirements.txt
pip install --no-index --find-links=${qwen3_vl_files%/}/wheelhouse/ torch==2.8.0 torchvision==0.23.0
pip install --no-index --find-links=${qwen3_vl_files%/}/wheelhouse/ --upgrade-strategy=only-if-needed -r requirements.txt
pip install --no-index --find-links=${qwen3_vl_files%/}/wheelhouse/ flash-attn==2.8.3 #+cu128torch2.8
pip install --no-index --find-links=${qwen3_vl_files%/}/wheelhouse/ ipdb


### Old method, tasks do not let you download
# #pip install torch==2.8.0 torchvision==0.23.0

# # get the absolute newest build, neeed .dev0 build for sana-video
# # pip install git+https://github.com/huggingface/diffusers

# # baseline dependencies
# pip install diffusers==0.31.0 transformers accelerate ftfy opencv-python ipdb
# pip install nvitop # useful


# # for self-forcing:
# pip install omegaconf einops easydict lmdb av
# # get flash attention from wheelhouse
# pip install --no-index --find-links=wheelhouse flashinfer-python
# # pip install --no-index --find-links=wheelhouse flash_attn

# # # Going to need to download flash_attn, save it to the wheel of dataset/qwen3_vl_files
# ## pip download --only-binary=:all: -d wheelhouse flash_attn
# ## ^ doesn't work with proxy.
# ## download this with pip from a repo
# ## I think its from, but I forgot how I uploaded it. wget and pip download don't seem to work anymore. That's how I had done it, but maybe just save it as a dataset now.
# #https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu128torch2.8-cp312-cp312-linux_x86_64.whl
# # pip install --no-index --find-links=wheelhouse flash_attn
# pip install wheelhouse/flash_attn-2.8.3+cu128torch2.8-cp312-cp312-linux_x86_64.whl

# pip install wandb

# #installation of vbench
# # apt install rustup
# # pip install vbench


############################  Testing  ############################
CUDA_VISIBLE_DEVICES="0,1" python - <<'PY'
import torch, ctypes, os
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.backends.cuda.is_built()", torch.backends.cuda.is_built())
print("torch.cuda.is_available()", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("name:", torch.cuda.get_device_name(0))
try:
    torch.cuda.init()
    print("CUDA init: OK")
except Exception as e:
    print("CUDA init error:", e)
PY

echo "WITHOUT CUDA_VISIBLE_DEVICES=\"0,1\""
python - <<'PY'
import torch, ctypes, os
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.backends.cuda.is_built()", torch.backends.cuda.is_built())
print("torch.cuda.is_available()", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("name:", torch.cuda.get_device_name(0))
try:
    torch.cuda.init()
    print("CUDA init: OK")
except Exception as e:
    print("CUDA init error:", e)
PY


# Use this python: /home/ma-user/miniconda3/bin/python
# either export PATH="$HOME/miniconda3/bin:$PATH" or just always use that, instead of just python

# cd fastvideo
# pip install -e .
# pip install vsa
# model path: /home/ma-user/work/dataset/john_wan2_1_t2v_1_3b_diffusers/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers_trimmed

# run example_infer.py

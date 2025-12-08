import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
    GivenFirstLatentCausalInferencePipeline
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
args = parser.parse_args()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)


# Load and inference on the baseline model
#==================================================================================

# import os
# os.environ["HF_HUB_OFFLINE"]="1"
# os.environ["HF_HOME"]="/home/ma-user/work/dataset"
# os.environ["TRANSFORMERS_CACHE"]="/home/ma-user/work/dataset"


# import torch

# # from diffusers import SanaImageToVideoPipeline, FlowMatchEulerDiscreteScheduler
# from diffusers import AutoencoderKLWan, WanPipeline#, FlowMatchEulerDiscreteScheduler
# from diffusers.utils import export_to_video

# model_id = "/home/ma-user/work/dataset/john_wan2_1_t2v_1_3b_diffusers/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers_trimmed/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b/"

# pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='cuda')
# prompt = "A gorilla happilly eats a banana."
# negative_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'


# if args.i2v:
#     assert not dist.is_initialized(), "I2V does not support distributed inference yet"
#     transform = transforms.Compose([
#         transforms.Resize((480, 832)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5])
#     ])
#     dataset = TextImagePairDataset(args.data_path, transform=transform)
# else:
#     dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
# num_prompts = len(dataset)

# if dist.is_initialized():
#     sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
# else:
#     sampler = SequentialSampler(dataset)
# dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)


# for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
#     idx = batch_data['idx'].item()
#     seed_idx = 0
#     prompts = batch_data['prompts']
#     negative_prompts = [negative_prompt]
#     frames = 1+4*20 # 81, 5s video
#     # frames = 1+4*16 # 4s video
#     # frames = 1+4*20 # 1s video

#     original_output = pipe(
#         prompt=prompts,
#         negative_prompt=negative_prompt,
#         height=480,
#         width=832,
#         num_frames=frames,
#         guidance_scale=3.0 # used in self-forcing config
#     ).frames[0]

# output_path = os.path.join(args.output_folder, f'baseline_{prompt[:100]}-{seed_idx}.mp4')
# from diffusers.utils import export_to_video
# export_to_video(original_output, output_path,fps=16)

import torchvision.io as io
video_path = r"/home/ma-user/work/algorithm/arvd_repos/new_training_self-forcing/videos/reinject_samples/baseline_A gorilla happilly eats a banana.-0.mp4"

# video_path = r"/home/ma-user/work/algorithm/arvd_repos/new_training_self-forcing/videos/reinject_samples/self_forcing_A gorilla happilly eats a banana.-0.mp4"

original_output, _, __ = io.read_video(video_path, pts_unit="sec")


#==================================================================================
#==================================================================================

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    print("load pipeline")
    
#     from wan.modules.causal_model import CausalWanModel
#     FOLDER_PATH="/home/ma-user/work/dataset/john_wan2_1_t2v_1_3B/models--Wan-AI--Wan2.1-T2V-1.3B/snapshots/37ec512624d61f7aa208f7ea8140a131f93afc9a"
#     causalpipe = CausalWanModel.from_pretrained(FOLDER_PATH)
    
    # john: use GivenFirstLatentCausalInferencePipeline
    pipeline = GivenFirstLatentCausalInferencePipeline(config, device=device)
    print("Done loading pipeline")
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    # john: send directly to gpu, because bandwidth is really low
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    pipeline.generator.load_state_dict(state_dict['generator' if not args.use_ema else 'generator_ema'])

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    if args.i2v:
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        prompts = [prompt] * args.num_samples

        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts'][0]
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] * args.num_samples
        else:
            prompts = [prompt] * args.num_samples
        
        # import ipdb;ipdb.set_trace()        
        #reencode the first 3 chunks latent chunks
        latent_input = torch.tensor(original_output[:12,],dtype=torch.bfloat16).unsqueeze(0).permute(0,4,1,2,3)
        latent_input = (latent_input / 255.0) * 2.0 - 1.0
        initial_latent = pipeline.vae.encode_to_latent(latent_input.to('cuda')).to(device=device, dtype=torch.bfloat16)
    
        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames-initial_latent.shape[1], 16, 60, 104], device=device, dtype=torch.bfloat16
        )
        
        # sampled_noise = torch.cat( (initial_latent,sampled_noise), axis=1)

    # Generate 81 frames
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        low_memory=low_memory,
    )
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        model = "regular" if not args.use_ema else "ema"
        for seed_idx in range(args.num_samples):
            # All processes save their videos
            if args.save_with_index:
                output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}_{model}.mp4')
            else:
                output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')
            write_video(output_path, video[seed_idx], fps=16)

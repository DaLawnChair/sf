"""
The purpose of this repostory is to make inference with the bidirectional model for video quality

The usage of this repo is a standardized version of the inference.py script for self-forcing to have a multi-instance video generation script that can be used for inference later.

# Requires:
* Paths set up (ie source get_paths.sh webstudio <repo path>
* environment sourcing

WARNING: To be tested... 
"""

# Set path so we can access the variables of the folder
import sys
import os
project_root = os.environ["TASK_RUN_REPO"]
sys.path.insert(0, project_root)

import argparse
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import math
from pipeline import (
    BidirectionalDiffusionInferencePipeline,
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.distributed import launch_distributed_job
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to the config file")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
    parser.add_argument("--data_path", type=str, help="Path to the dataset")
    parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
    parser.add_argument("--output_folder", type=str, help="Output folder")
    parser.add_argument("--num_output_frames", type=int, default=21,
                        help="Number of overlap frames between sliding windows")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
    parser.add_argument("--save_with_index", action="store_true",
                        help="Whether to save the video using the index or prompt as the filename")
    parser.add_argument("--use_bidirectional", action="store_true",
                        help="Whether to use the bidirectional model")



    args = parser.parse_args()
    torch.set_grad_enabled(False)

    # load config
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load(os.path.join(project_root,"configs/default_config.yaml"))
    config = OmegaConf.merge(default_config, config)

    # launch as distributed job
    if "LOCAL_RANK" in os.environ:
        launch_distributed_job()
        device = torch.cuda.current_device()
        local_rank = dist.get_rank() 
        set_seed(args.seed + local_rank)
    else:
        device = torch.cuda.current_device()
        world_size = 1
        local_rank = 0
        set_seed(args.seed)

    # Initialize pipeline
    pipeline = BidirectionalDiffusionInferencePipeline(config, device=device)    
    pipeline = pipeline.to(torch.cuda.current_device())

    # access dataset
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)

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



    import time 
    total_duration = []

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    local_rank = dist.get_rank() if dist.is_initialized() else 0

    for index in tqdm(range(int(math.ceil(len(dataset) / world_size))), disable=local_rank != 0):
        prompt_index = index * world_size + local_rank
        batch_data = dataset[prompt_index]

        # For dataset batch_size=1, the batch_data is already a single item, but in a batch container
        # Unpack the batch data for convenience
        if isinstance(batch_data, dict):
            batch = batch_data
        elif isinstance(batch_data, list):
            batch = batch_data[0]  # First (and only) item in the batch

        all_video = []
        num_generated_frames = 0  # Number of generated (latent) frames


        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts']

        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] * args.num_samples
        else:
            prompts = [prompt] * args.num_samples
        initial_latent = None

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.float32
        )
        
        start_time = time.time()
        # Generate 81 frames
        if args.use_bidirectional:
            video, latents= pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=True # need latents back
            )
        else:
            video, latents = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=True, # need latents back
                initial_latent=initial_latent,
                # low_memory=low_memory,
            )
        end_time = time.time()
        duration = end_time - start_time
        total_duration.append(duration)

        
        if args.view_videos:
            current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
            all_video.append(current_video)

            # Final output video
            video = 255.0 * torch.cat(all_video, dim=1)

            # Clear VAE cache
            pipeline.vae.model.clear_cache()
            
        # Save the video if the current prompt is not a dummy prompt                
        output_path = os.path.join(args.output_folder, f'{prompt_index}.pt')

        if args.view_videos:
            stored_data = {
                "noise": sampled_noise.cpu(),
                "video": video.cpu(),
            }
        else:
            stored_data = {
                "noise": sampled_noise.cpu(),
                "video": latents.cpu(),
            }
        torch.save({prompts[0]: stored_data}, output_path)

    if dist.is_initialized():
        dist.barrier()

if __name__=='__main__':
    main()
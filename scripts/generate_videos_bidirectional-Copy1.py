



"""
This file will generate videos using the original wan2.1-t2v-1.3B model using torchrun.

This is a modification of the scripts/generate_ode_pairs.py file

* Added sys path
* sending args to init_model() to change the shift 
* add timeshift_scale, guidance_scale to the args.
* generating videos instead of latents
"""

# === sys.path fix ===
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# =====================

from utils.distributed import launch_distributed_job
from utils.dataset import TextDataset
import torch.distributed as dist
from tqdm import tqdm
import argparse
import torch
import math

from pipeline import BidirectionalDiffusionInferencePipeline
from einops import rearrange
from utils.misc import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--caption_path", type=str)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--timeshift_scale", type=float, default=8.0)

    # These are not meant to be changed but kept for API compatibility
    parser.add_argument("--num_train_timestep", type=float, default=1000)
    parser.add_argument("--denoising_step_list", type=float, default=0)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
            "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
            "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # ============================================================
    # Distributed + device setup
    # ============================================================
    # Let the repo's helper set up process group, env vars, etc.
    launch_distributed_job()

    # Use LOCAL_RANK to determine device explicitly
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # Fallback for single-process debug
        local_rank = 0

    # FORCE this process to use its GPU only
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(
        f"[Rank {global_rank}] LOCAL_RANK={local_rank}, "
        f"device={device}, world_size={world_size}"
    )

    set_seed(args.seed + global_rank)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dist.barrier()

    # ============================================================
    # Build pipeline on the correct device
    # ============================================================
    # IMPORTANT: pass `device` to the pipeline and move modules explicitly
    pipeline = BidirectionalDiffusionInferencePipeline(args, device=device)
    pipeline.to(device)  # should internally move generator, vae, text_encoder

    # Sanity check: where are the modules actually located?
    print(
        f"[Rank {global_rank}] generator on "
        f"{next(pipeline.generator.parameters()).device}, "
        f"text encoder on {next(pipeline.text_encoder.parameters()).device}, "
        f"vae on {next(pipeline.vae.parameters()).device}"
    )

    dist.barrier()

    # ============================================================
    # Dataset & output
    # ============================================================
    dataset = TextDataset(args.caption_path)
    os.makedirs(args.output_folder, exist_ok=True)

    # Split dataset indices across ranks
    num_samples_per_rank = int(math.ceil(len(dataset) / world_size))
    for index in tqdm(
        range(num_samples_per_rank),
        disable=(global_rank != 0),
        desc=f"Rank {global_rank} generating",
        ):
        prompt_index = index * world_size + global_rank
        if prompt_index >= len(dataset):
            continue

        prompt = dataset[prompt_index]
        prompts = prompt["prompts"]

        # ========================================================
        # Noise on the *correct* device
        # ========================================================
        noise = torch.randn(
            [1, 21, 16, 60, 104], dtype=torch.float32, device=device
        )
        print(
            f"[Rank {global_rank}] prompt_index={prompt_index}, "
            f"noise.device={noise.device}"
        )

        import copy

        pure_noise = copy.deepcopy(noise.cpu())

        # ========================================================
        # Inference on the correct device (hard guard)
        # ========================================================
        # If anything inside the pipeline uses default CUDA device or "cuda",
        # this context forces it to be our rank's device.
        with torch.cuda.device(device):
            # Optionally, also enforce inside pipeline:
            # (if you can modify it, at the top of inference do
            #  torch.cuda.set_device(self.device.index))
            video = pipeline.inference(
                noise=noise,
                text_prompts=prompts,
                return_latents=False,
            )

        # video: [B, T, C, H, W]
        # We only generate one chunk per prompt here, so no need to concat
        current_video = rearrange(video, "b t c h w -> b t h w c").cpu()
        all_video = [current_video]  # if you later extend to multiple chunks, append

        # Final output video: concat along time dimension (here it's just one)
        video_out = 255.0 * torch.cat(all_video, dim=1)

        stored_data = {
            "noise": pure_noise.detach(),
            "video": video_out.detach(),
        }

        # prompt is a dict; use the string prompt key
        prompt_str = prompt["prompts"]
        torch.save(
            {prompt_str: stored_data},
            os.path.join(args.output_folder, f"{prompt_index:05d}.pt"),
        )

        # Clear VAE cache via pipeline
        if hasattr(pipeline.vae, "model") and hasattr(
            pipeline.vae.model, "clear_cache"
        ):
            pipeline.vae.model.clear_cache()

    dist.barrier()


if __name__ == "__main__":
    main()




# """

# This file will generate videos using the original wan2.1-t2v-1.3B model using torchrun.

# This is a modification of the scripts/generate_ode_pairs.py file

# * Added sys path
# * sending args to init_model() to change the shift 
# * add timeshift_scale, guidance_scale to the args.
# * generating videos instead of latents


# """


# # ===
# #john: python scripts/generate_ode_pairs.py leads to import errors. Fix them
# import sys
# import os
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, project_root)
# # ===


# from utils.distributed import launch_distributed_job
# from utils.scheduler import FlowMatchScheduler
# from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
# from utils.dataset import TextDataset
# import torch.distributed as dist
# from tqdm import tqdm
# import argparse
# import torch
# import math
# import os


# from pipeline import BidirectionalDiffusionInferencePipeline


# from einops import rearrange # need this

# from utils.misc import set_seed


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--local_rank", type=int, default=-1)
#     parser.add_argument("--output_folder", type=str)
#     parser.add_argument("--caption_path", type=str)
#     parser.add_argument("--guidance_scale", type=float, default=6.0)
#     parser.add_argument("--timeshift_scale", type=float, default=8.0) # add timeshift scale
    
    
#     # these are not made to be changed
#     parser.add_argument("--num_train_timestep", type=float, default=1000)
#     parser.add_argument("--denoising_step_list", type=float, default=0)
#     parser.add_argument("--negative_prompt", type=str, default='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走')
#     parser.add_argument("--seed", type=int, default=0)
    
#     args = parser.parse_args()
    
   

#     torch.set_grad_enabled(False)
 
#     # =================================
    
#     # launch_distributed_job()
#     launch_distributed_job()
    
# #     print(os.environ["LOCAL_RANK"])
# #     print(dist.get_world_size())
# #     print(torch.cuda.current_device())


#     device = torch.cuda.current_device()

#     print("device",device)
    
#     torch.set_grad_enabled(False)
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True
    
#     torch.cuda.set_device(device)
    
#     dist.barrier()
#     pipeline = BidirectionalDiffusionInferencePipeline(args, device=device)
    
#     pipeline.generator.to(device)
#     pipeline.vae.to(device)
#     pipeline.text_encoder.to(device)
    
#     pipeline.to(device)
    
#     dist.barrier()    

#     dataset = TextDataset(args.caption_path)

#     # if global_rank == 0:
#     os.makedirs(args.output_folder, exist_ok=True)
    
    
    
#     print(
#         f"[Rank: {dist.get_rank()} generator on {next(pipeline.generator.parameters()).device}]",
#         f"[text encoder on {next(pipeline.text_encoder.parameters()).device}]",
#         f"[vae on {next(pipeline.vae.parameters()).device}]",
#     )
          

#     for index in tqdm(range(int(math.ceil(len(dataset) / dist.get_world_size()))), disable=dist.get_rank() != 0):

#         prompt_index = index * dist.get_world_size() + dist.get_rank()
#         if prompt_index >= len(dataset):
#             continue
            
#         prompt = dataset[prompt_index]

#         prompts = prompt["prompts"]
#         noise = torch.randn(
#             [1, 21, 16, 60, 104], dtype=torch.float32, device=device
#         )
#         print("noise.device inside of generate_videos.py", noise.device)
        
#         # import copy
#         # pure_noise = copy.deepcopy(noise.cpu())

#         # for progress_id, t in enumerate(tqdm(pipeline.generator.scheduler.timesteps)):
            
            
#         with torch.cuda.device(device):
#             video = pipeline.inference(
#                 noise=noise,
#                 text_prompts=prompts,
#                 return_latents=False, # don't send latents back
#                 # low_memory=low_memory,
#             )

            
#         current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
#         all_video.append(current_video)

#         # Final output video
#         video = 255.0 * torch.cat(all_video, dim=1)
    
#         stored_data = {
#             'noise': pure_noise.cpu().detach(),
#             'video': video.cpu().detach()
#         }

#         torch.save(
#             {prompt['prompts']: stored_data}, # john: prompt is a dict, use string
#             os.path.join(args.output_folder, f"{prompt_index:05d}.pt")
#         )
#         vae.model.clear_cache()


#     dist.barrier()


# if __name__ == "__main__":        
#     main()

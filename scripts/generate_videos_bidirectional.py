"""
Added sys path

sending args to init_model() to change the shift 

add timeshift_scale to the args.

"""


# ===
#john: python scripts/generate_ode_pairs.py leads to import errors. Fix them
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# ===


from utils.distributed import launch_distributed_job
from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
from utils.dataset import TextDataset
import torch.distributed as dist
from tqdm import tqdm
import argparse
import torch
import math
import os


# from pipeline import BidirectionalDiffusionInferencePipeline
from wan import WanT2V
from wan.configs import t2v_1_3B
from einops import rearrange
from utils.misc import set_seed


# imports just for the generate() function
# ===========================

import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from wan.distributed.fsdp import shard_model
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
# ===========================
# this just does the same as WanT2V.generate(), but returns the noise
def return_noise_generate(self,
             input_prompt,
             size=(1280, 720),
             frame_num=81,
             shift=5.0,
             sample_solver='unipc',
             sampling_steps=50,
             guide_scale=5.0,
             n_prompt="",
             seed=-1,
             offload_model=True):
    r"""
    Generates video frames from text prompt using diffusion process.

    Args:
        input_prompt (`str`):
            Text prompt for content generation
        size (tupele[`int`], *optional*, defaults to (1280,720)):
            Controls video resolution, (width,height).
        frame_num (`int`, *optional*, defaults to 81):
            How many frames to sample from a video. The number should be 4n+1
        shift (`float`, *optional*, defaults to 5.0):
            Noise schedule shift parameter. Affects temporal dynamics
        sample_solver (`str`, *optional*, defaults to 'unipc'):
            Solver used to sample the video.
        sampling_steps (`int`, *optional*, defaults to 40):
            Number of diffusion sampling steps. Higher values improve quality but slow generation
        guide_scale (`float`, *optional*, defaults 5.0):
            Classifier-free guidance scale. Controls prompt adherence vs. creativity
        n_prompt (`str`, *optional*, defaults to ""):
            Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
        seed (`int`, *optional*, defaults to -1):
            Random seed for noise generation. If -1, use random seed.
        offload_model (`bool`, *optional*, defaults to True):
            If True, offloads models to CPU during generation to save VRAM

    Returns:
        torch.Tensor:
            Generated video frames tensor. Dimensions: (C, N H, W) where:
            - C: Color channels (3 for RGB)
            - N: Number of frames (81)
            - H: Frame height (from size)
            - W: Frame width from size)
    """
    # preprocess
    F = frame_num
    target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                    size[1] // self.vae_stride[1],
                    size[0] // self.vae_stride[2])

    seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                        (self.patch_size[1] * self.patch_size[2]) *
                        target_shape[1] / self.sp_size) * self.sp_size

    if n_prompt == "":
        n_prompt = self.sample_neg_prompt
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)

    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context = self.text_encoder([input_prompt], torch.device('cpu'))
        context_null = self.text_encoder([n_prompt], torch.device('cpu'))
        context = [t.to(self.device) for t in context]
        context_null = [t.to(self.device) for t in context_null]

    noise = [
        torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=seed_g)
    ]

    pure_noise = noise[0].cpu().clone().detach().unsqueeze(0) # john: save noise
    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, 'no_sync', noop_no_sync)

    # evaluation mode
    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        # sample videos
        latents = noise

        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}

        for _, t in enumerate(tqdm(timesteps)):
            latent_model_input = latents
            timestep = [t]

            timestep = torch.stack(timestep)

            self.model.to(self.device)
            noise_pred_cond = self.model(
                latent_model_input, t=timestep, **arg_c)[0]
            noise_pred_uncond = self.model(
                latent_model_input, t=timestep, **arg_null)[0]

            noise_pred = noise_pred_uncond + guide_scale * (
                noise_pred_cond - noise_pred_uncond)

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g)[0]
            latents = [temp_x0.squeeze(0)]

        x0 = latents
        if offload_model:
            self.model.cpu()
        # if self.rank == 0:
        #     videos = self.vae.decode(x0)



    del noise, latents
    del sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        
        
    videos = self.vae.decode(x0)
    videos = torch.stack(videos)
    return pure_noise, videos # return noise and video
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--caption_path", type=str)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--timeshift_scale", type=float, default=8.0) # add timeshift scale
    
    # These are not meant to be changed but kept for API compatibility
    parser.add_argument("--num_train_timestep", type=float, default=1000)
    parser.add_argument("--denoising_step_list", type=float, default=0)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
    )
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    
    
    # launch_distributed_job()
    launch_distributed_job()

    device = torch.cuda.current_device()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # model, encoder, scheduler, unconditional_dict = init_model(args,device=device)
    
    with torch.cuda.device(device):
#         pipeline = BidirectionalDiffusionInferencePipeline(args,device=device)
#         pipeline.to(device)
        
#         pipeline.generator = pipeline.generator.to(device).to(torch.float32)
#         pipeline.text_encoder = pipeline.text_encoder.to(device).to(torch.float32)
#         pipeline.vae = pipeline.vae.to(device).to(torch.float32)


        WanT2V.generate = return_noise_generate # monkeypatch for the new generate
        config = t2v_1_3B
        checkpoint_dir = f"{os.environ["DATASET_PATH"]}john_wan2_1_t2v_1_3B/models--Wan-AI--Wan2.1-T2V-1.3B/snapshots/37ec512624d61f7aa208f7ea8140a131f93afc9a"
        model = WanT2V(
            config,
            checkpoint_dir,
            device_id=device,
            rank=int(os.environ["LOCAL_RANK"]),
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=False,
        )
                
        model.sample_neg_prompt = args.negative_prompt
        
            
    print("===============================================",
        os.environ['LOCAL_RANK'],
        # 'device:', device,
        # "pipeline.generator.model.device", pipeline.generator.model.device,
        # # "pipeline.vae.device", pipeline.vae.device,
        # "pipeline.text_encoder.device", pipeline.text_encoder.device,
        # "pipeline.text_encoder.text_encoder.blocks[0].attn.q.weight.device", pipeline.text_encoder.text_encoder.blocks[0].attn.q.weight.device,
          
        "==============================================="
    )

    dist.barrier()
    dataset = TextDataset(args.caption_path)

    # if global_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

    for index in tqdm(range(int(math.ceil(len(dataset) / dist.get_world_size()))), disable=dist.get_rank() != 0):
        prompt_index = index * dist.get_world_size() + dist.get_rank()
        if prompt_index >= len(dataset):
            continue
        prompt = dataset[prompt_index]

        # conditional_dict = encoder(text_prompts=prompt)

#         latents = torch.randn(
#             [1, 21, 16, 60, 104], dtype=torch.float32, device=device
#         )
        
#         latents = latents.to(device)
        
#         pure_noise = latents.clone()


        
        args.timeshift # default 5.0
        args.guidance_scale # default 5.0
        pure_noise, video = model.generate(
                 input_prompt = prompt['prompts'],
                 size=(832, 480),
                 frame_num=81,
                 shift=args.timeshift, # default 5.0
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=args.guidance_scale, # default 5.0
                 n_prompt=args.negative_prompt,
                 seed=0, # default is -1, but we want 0 for fixed results
                 offload_model=False)
                        
        # video = pipeline.inference(
        #     noise=pure_noise,
        #     text_prompts=[prompt['prompts']]
        # )

            
        # for progress_id, t in enumerate(tqdm(scheduler.timesteps)):
        #     timestep = t * \
        #         torch.ones([1, 21], device=device, dtype=torch.float32)


            # noisy_input.append(latents)

#             _, x0_pred_cond = model(
#                 latents, conditional_dict, timestep
#             )

#             _, x0_pred_uncond = model(
#                 latents, unconditional_dict, timestep
#             )

#             x0_pred = x0_pred_uncond + args.guidance_scale * (
#                 x0_pred_cond - x0_pred_uncond
#             )

#             flow_pred = model._convert_x0_to_flow_pred(
#                 scheduler=scheduler,
#                 x0_pred=x0_pred.flatten(0, 1),
#                 xt=latents.flatten(0, 1),
#                 timestep=timestep.flatten(0, 1)
#             ).unflatten(0, x0_pred.shape[:2])

#             latents = scheduler.step(
#                 flow_pred.flatten(0, 1),
#                 scheduler.timesteps[progress_id] * torch.ones(
#                     [1, 21], device=device, dtype=torch.long).flatten(0, 1),
#                 latents.flatten(0, 1)
#             ).unflatten(dim=0, sizes=flow_pred.shape[:2])

#         noisy_input.append(latents)

#         noisy_inputs = torch.stack(noisy_input, dim=1)

#         noisy_inputs = noisy_inputs[:, [0, 12, 24, 36, -1]]

#         stored_data = noisy_inputs

        
        current_video = video
        # current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        current_video = 255.0 * torch.cat([current_video],dim=1)

        
        print("current_video.shape",current_video.shape)
        stored_data = {
            'noise':pure_noise,
            'video':current_video.cpu()
        }
        torch.save(
            {prompt['prompts']: stored_data}, # john: prompt is a dict, use string
            os.path.join(args.output_folder, f"{prompt_index:06d}.pt")
        )

    dist.barrier()


if __name__ == "__main__":
    main()

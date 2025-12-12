



"""

This file will generate videos using the original wan2.1-t2v-1.3B model using torchrun.

This is a modification of the scripts/generate_ode_pairs.py file

* Added sys path
* sending args to init_model() to change the shift 
* add timeshift_scale, guidance_scale to the args.
* generating videos instead of latents


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
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.dataset import TextDataset
import torch.distributed as dist
from tqdm import tqdm
import argparse
import torch
import math
import os


from einops import rearrange # need this


def init_model(args, device):
    model = WanDiffusionWrapper().to(device).to(torch.float32)
    encoder = WanTextEncoder().to(device).to(torch.float32)
    vae = WanVAEWrapper().to(device).to(torch.float32)
    model.model.requires_grad_(False)
    
    
    scheduler = FlowMatchScheduler(
        shift=args.timeshift_scale, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(num_inference_steps=50, denoising_strength=1.0) # changed from 48 to 50, denoising_strength is default
    scheduler.sigmas = scheduler.sigmas.to(device)

    
    
    sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

    unconditional_dict = encoder(
        text_prompts=[sample_neg_prompt]
    )

    return model, encoder, scheduler, vae, unconditional_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--caption_path", type=str)
    # parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--timeshift_scale", type=float, default=8.0) # add timeshift scale
    
    # these are not made to be changed
    # parser.add_argument("--num_train_timestep", type=float, default=1000)
    # parser.add_argument("--denoising_step_list", type=float, default=0)
    # parser.add_argument("--negative_prompt", type=str, default='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走')
    
    
    
    args = parser.parse_args()

    # launch_distributed_job()
    launch_distributed_job()

    device = torch.cuda.current_device()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, encoder, scheduler, vae, unconditional_dict = init_model(args,device=device)
    pipeline = init_model(args,device=device)
    

    dataset = TextDataset(args.caption_path)

    # if global_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

    for index in tqdm(range(int(math.ceil(len(dataset) / dist.get_world_size()))), disable=dist.get_rank() != 0):

        prompt_index = index * dist.get_world_size() + dist.get_rank()
        if prompt_index >= len(dataset):
            continue
            
        prompt = dataset[prompt_index]

        conditional_dict = encoder(text_prompts=prompt)
        print(prompt_index)
        print(prompt)

        latents = torch.randn(
            [1, 21, 16, 60, 104], dtype=torch.float32, device=device
        )
        
        import copy
        pure_noise = copy.deepcopy(latents.cpu())

        for progress_id, t in enumerate(tqdm(scheduler.timesteps)):
            
            timestep = t * \
                torch.ones([1, 21], device=device, dtype=torch.float32)

            # flow_pred_cond, _ = model(latents, conditional_dict, timestep)
            # flow_pred_uncond, _ = model(latents, unconditional_dict, timestep)
            _, x0_pred_cond = model(latents, conditional_dict, timestep)
            _, x0_pred_uncond = model(latents, unconditional_dict, timestep)
            
            x0_pred = x0_pred_uncond + args.guidance_scale * (
                x0_pred_cond - x0_pred_uncond)
            
#             flow_pred = model._convert_x0_to_flow_pred(
#                 scheduler = scheduler,
#                 x0_pred=x0_pred.flatten(0,1),
#                 xt=latents.flatten(0,1),
#                 timestep=timestep.flatten(0,1),
#             ).unflatten(0,x0_pred.shape[:2])
            
            
            # latents = scheduler.step(
            #     flow_pred.flatten(0,1),
            #     timestep,
            #     latents.unsqueeze(0),
            #     ).unflatten(dim=0,sizes=flow_pred.shape[:2])
            
            latents = scheduler.step(
                x0_pred.flatten(0,1),
                timestep,
                latents, # this line doesn't do anything [][]
            )[0]
            
            
            # flow_pred = flow_pred_uncond + args.guidance_scale * (
            #     flow_pred_cond - flow_pred_uncond)

            # latents = scheduler.step(
            #     flow_pred.unsqueeze(0),
            #     timestep,
            #     latents.unsqueeze(0),
            #     )[0] 


        # video = latents.squeeze(0)
        video = latents
        print('video1',video.shape)
 
        video = vae.decode_to_pixel(video, use_cache=False)
        print('video_decode',video.shape)
        
        video = (video * 0.5 + 0.5).clamp(0, 1)
        
        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        print('current_video_rearrance',current_video.shape)
        
        all_video = [current_video]
        video = 255.0 * torch.cat(all_video, dim=1)
        
    
        stored_data = {
            'noise': pure_noise.cpu().detach(),
            'video': video.cpu().detach()
        }

        # torch.save(
        #     {prompt['prompts']: stored_data.cpu().detach()}, # john: prompt is a dict, use string
        #     os.path.join(args.output_folder, f"{prompt_index:05d}.pt")
        # )
        torch.save(
            {prompt['prompts']: stored_data}, # john: prompt is a dict, use string
            os.path.join(args.output_folder, f"{prompt_index:05d}.pt")
        )
        vae.model.clear_cache()


    dist.barrier()


if __name__ == "__main__":
    main()





# """
# Added sys path

# sending args to init_model() to change the shift 

# add timeshift_scale to the args.



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


# from einops import rearrange # need this


# from pipeline import BidirectionalDiffusionInferencePipeline

# def init_model(args, device):
#     # model = WanDiffusionWrapper().to(device).to(torch.float32)
#     # encoder = WanTextEncoder().to(device).to(torch.float32)
#     # vae = WanVAEWrapper().to(device).to(torch.float32)
#     # model.model.requires_grad_(False)

    
#     # timeshift_scale = args.timeshift_scale

    
    
#     scheduler = FlowMatchScheduler(
#         shift=args.timeshift_scale, sigma_min=0.0, extra_one_step=True)
#     scheduler.set_timesteps(num_inference_steps=50, denoising_strength=1.0) # changed from 48 to 50, denoising_strength is default
#     scheduler.sigmas = scheduler.sigmas.to(device)

    
    
#     pipeline = BidirectionalDiffusionInferencePipeline(args,device=device)
#     pipeline.generator.scheduler = scheduler
#     pipeline.denoising_step_list = scheduler.timesteps
    
#     pipeline.to(device)
    
    
    
#     sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

#     # unconditional_dict = encoder(
#     #     text_prompts=[sample_neg_prompt]
#     # )

#     # return model, encoder, scheduler, vae, unconditional_dict
#     return pipeline


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
    
    
    
#     args = parser.parse_args()

#     # launch_distributed_job()
#     launch_distributed_job()

#     device = torch.cuda.current_device()

#     torch.set_grad_enabled(False)
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True

#     # model, encoder, scheduler, vae, unconditional_dict = init_model(args,device=device)
#     pipeline = init_model(args,device=device)
    

#     dataset = TextDataset(args.caption_path)

#     # if global_rank == 0:
#     os.makedirs(args.output_folder, exist_ok=True)

#     for index in tqdm(range(int(math.ceil(len(dataset) / dist.get_world_size()))), disable=dist.get_rank() != 0):

#         prompt_index = index * dist.get_world_size() + dist.get_rank()
#         if prompt_index >= len(dataset):
#             continue
#         prompt = dataset[prompt_index]

#         # conditional_dict = encoder(text_prompts=prompt)

#         latents = torch.randn(
#             [1, 21, 16, 60, 104], dtype=torch.float32, device=device
#         )
        
#         import copy
#         pure_noise = copy.deepcopy(latents.cpu())

#         for progress_id, t in enumerate(tqdm(pipeline.generator.scheduler.timesteps)):
            
#             video = pipeline.inference(pure_noise, [prompt['prompts']])
            
            
# #             timestep = t * \
# #                 torch.ones([1, 21], device=device, dtype=torch.float32)

            
# #             timestep = t * torch.ones([latents.shape[0], 21], device=latents.device, dtype=torch.float32)

# #             flow_pred_cond, _ = model(latents, conditional_dict, timestep)

# #             flow_pred = flow_pred_uncond + args.guidance_scale * (
# #                 flow_pred_cond - flow_pred_uncond)

# #             temp_x0 = scheduler.step(
# #                 flow_pred.unsqueeze(0),
# #                 timestep,
# #                 latents.unsqueeze(0),
# #                 )[0]
# #             latents = temp_x0.squeeze(0)
            
# #             _, x0_pred_cond = model(
# #                 latents, conditional_dict, timestep
# #             )

# #             _, x0_pred_uncond = model(
# #                 latents, unconditional_dict, timestep
# #             )

# #             x0_pred = x0_pred_uncond + args.guidance_scale * (
# #                 x0_pred_cond - x0_pred_uncond
# #             )

# #             flow_pred = model._convert_x0_to_flow_pred(
# #                 scheduler=scheduler,
# #                 x0_pred=x0_pred.flatten(0, 1),
# #                 xt=latents.flatten(0, 1),
# #                 timestep=timestep.flatten(0, 1)
# #             ).unflatten(0, x0_pred.shape[:2])

# #             latents = scheduler.step(
# #                 flow_pred.flatten(0, 1),
# #                 scheduler.timesteps[progress_id] * torch.ones(
# #                     [1, 21], device=device, dtype=torch.long).flatten(0, 1),
# #                 latents.flatten(0, 1)
# #             ).unflatten(dim=0, sizes=flow_pred.shape[:2])

#         video = latents
 
#         video = vae.decode_to_pixel(video, use_cache=False)
#         video = (video * 0.5 + 0.5).clamp(0, 1)
#         current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
#         all_video = [current_video]
#         video = 255.0 * torch.cat(all_video, dim=1)
        
#         vae.model.clear_cache()
    
#         stored_data = {
#             'noise': pure_noise.cpu().detach(),
#             'video': video.cpu().detach()
#         }

#         # torch.save(
#         #     {prompt['prompts']: stored_data.cpu().detach()}, # john: prompt is a dict, use string
#         #     os.path.join(args.output_folder, f"{prompt_index:05d}.pt")
#         # )
#         torch.save(
#             {prompt['prompts']: stored_data}, # john: prompt is a dict, use string
#             os.path.join(args.output_folder, f"{prompt_index:05d}.pt")
#         )

#     dist.barrier()


# if __name__ == "__main__":
#     main()

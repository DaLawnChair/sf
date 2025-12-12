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


from pipeline import BidirectionalDiffusionInferencePipeline
from einops import rearrange
from utils.misc import set_seed


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
        default=(
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
            "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
            "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        ),
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
        pipeline = BidirectionalDiffusionInferencePipeline(args,device=device)
        pipeline.to(device)
        
        pipeline.generator = pipeline.generator.to(device).to(torch.float32)
        pipeline.text_encoder = pipeline.text_encoder.to(device).to(torch.float32)
        pipeline.vae = pipeline.vae.to(device).to(torch.float32)
    
    
    print("===============================================",
        os.environ['LOCAL_RANK'],
        'device:', device,
        "pipeline.generator.model.device", pipeline.generator.model.device,
        # "pipeline.vae.device", pipeline.vae.device,
        "pipeline.text_encoder.device", pipeline.text_encoder.device,
        "pipeline.text_encoder.text_encoder.blocks[0].attn.q.weight.device", pipeline.text_encoder.text_encoder.blocks[0].attn.q.weight.device,
          
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

        latents = torch.randn(
            [1, 21, 16, 60, 104], dtype=torch.float32, device=device
        )
        
        latents = latents.to(device)
        
        pure_noise = latents.clone()

        noisy_input = []

            
        video = pipeline.inference(
            noise=pure_noise,
            text_prompts=[prompt['prompts']]
        )

            
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


        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        current_video = 255.0 * torch.cat([current_video],dim=1)

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

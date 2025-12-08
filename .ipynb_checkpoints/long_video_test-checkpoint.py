import os
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
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

import json # for tracking results

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

batch_size = 1
# video_lengths = [21*i for i in range(1,60//5,2)] # technically wrong, but needed for passing value through

video_lengths = [21, 63, 126, 189, 252]
# represents 5, 15, 30, 45, 60 second generation durations inside of latent frame form
results = {}
for video_length in video_lengths:
    real_video_length = (video_length - (video_length//21)) //4
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


    if 'bidirectional' in args.config_path:
        from pipeline import (
            BidirectionalDiffusionInferencePipeline,
            BidirectionalInferencePipeline,
        )
        if 'diffusion' in args.config_path:
            pipeline = BidirectionalDiffusionInferencePipeline(config, device=device)
        else:
            pipeline = BidirectionalInferencePipeline(config, device=device)
    else:
        # Initialize pipeline
        if hasattr(config, 'denoising_step_list'):
            # Few-step inference
            local_attn_size = 21 # window if 81 frames
            pipeline = CausalInferencePipeline(config, device=device, local_attn_size=local_attn_size)
        else:
            # Multi-step diffusion inference
            pipeline = CausalDiffusionInferencePipeline(config, device=device)
    print("type(pipeline)",type(pipeline))

    if args.checkpoint_path:
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        pipeline.generator.load_state_dict(state_dict['generator' if not args.use_ema else 'generator_ema'])

    # pipeline = pipeline.to(dtype=torch.bfloat16)
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    else:
        pipeline.text_encoder.to(device=gpu)
    pipeline.generator.to(device=gpu)
    pipeline.vae.to(device=gpu)

    # import ipdb;ipdb.set_trace()
    pipeline.to(device="cuda", dtype=torch.bfloat16) # john: add in this line to be consistent with causvid
    
    
    # john: preallocate the memory, so that 6GiB of memory is taken out of reserve
    pipeline._initialize_kv_cache(1,torch.bfloat16,device)
    pipeline._initialize_crossattn_cache(1,torch.bfloat16,device)

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("Post initalization, emptying cache")
    print("max memory reserved", round(torch.cuda.max_memory_reserved()/ (1024**3),4) )
    print("max memory allocated", round(torch.cuda.max_memory_allocated()/ (1024**3),4) )

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
    

    # limit the # of prompts to 10 samples for each batch size for testing
    dataset.prompt_list = dataset.prompt_list[:batch_size*10]
    num_prompts = len(dataset)
    print(f"Number of prompts: {num_prompts}")

    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=False)

    args.num_samples = batch_size
    args.num_output_frames = video_length
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


    import time 
    total_duration = []
    idx = 0 # dummy variable
    for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
        # idx = batch_data['idx'].item()

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
            initial_latent = None


            sampled_noise = torch.randn(
                [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
            )
        print(f"========== Prompt #{i+1} ==========")
        start_time = time.time()
        # Generate 81 frames
        if 'bidirectional' in args.config_path:
            video = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts
            )

        else:

            video = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=False, # don't send latents back
                initial_latent=initial_latent,
                # low_memory=low_memory,
            )
        end_time = time.time()
        duration = end_time - start_time
        total_duration.append(duration)
        print("duration",duration)
        print("video shape", video.shape)
        
        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        all_video.append(current_video)
        
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
                    output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed_idx}-{args.num_output_frames}.mp4')

                # save the video for bs=1, otherwise only save the first batch
                if batch_size==1:
                    write_video(output_path, video[seed_idx], fps=16)
                else:
                    if i==0:
                        write_video(output_path, video[seed_idx], fps=16)
    # keep track of results
    results[real_video_length] = {
        "total_duration": sum(total_duration),
        "average_duration_per_video": sum(total_duration)/len(total_duration),
        "actual_durations": total_duration,
        "max_memory_reserved": round(torch.cuda.max_memory_reserved()/ (1024**3),4),
        "max_memory_allocated": round(torch.cuda.max_memory_allocated()/ (1024**3),4)
        
    }

    print(f"RESULTS FOR real_video_length={real_video_length}")
    print("===================================")
    print("total_duration=",results[real_video_length]["total_duration"])
    print("average_duration_per_video=",results[real_video_length]["average_duration_per_video"])
    print("actual_durations=",results[real_video_length]["actual_durations"])
    print("max_memory_reserved=",results[real_video_length]["max_memory_reserved"])
    print("max_memory_allocated=",results[real_video_length]["max_memory_allocated"])
    with open(os.path.join(args.output_folder, f'long_generation_results.txt'), 'a') as f:
        json.dump(results, f, indent=4)

    torch.cuda.reset_max_memory_allocated() # reset the memory for next run
    torch.cuda.memory.reset_peak_memory_stats() # reset the memory for next run



    # reset for a new run    
    del pipeline
    del sampled_noise

    import gc 
    gc.collect()
    torch.cuda.empty_cache()
    # import ipdb;ipdb.set_trace()
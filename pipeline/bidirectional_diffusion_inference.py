"""
Has updates
"""
from tqdm import tqdm
from typing import List
import torch

from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class BidirectionalDiffusionInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=False) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize scheduler
        self.num_train_timesteps = args.num_train_timestep
        self.sampling_steps = 50
        self.sample_solver = 'unipc'
        self.shift = 8.0

        self.args = args

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        return_latents=False
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """

        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )

        latents = noise

        self.latent_history = []
        
        sample_scheduler = self._initialize_sample_scheduler(noise)
        for _, t in enumerate(tqdm(sample_scheduler.timesteps)):
            latent_model_input = latents
            timestep = t * torch.ones([latents.shape[0], 21], device=noise.device, dtype=torch.float32)

            flow_pred, _ = self.generator(latent_model_input, conditional_dict, timestep)

            flow_pred_cond, _ = self.generator(latent_model_input, conditional_dict, timestep)
            flow_pred_uncond, _ = self.generator(latent_model_input, unconditional_dict, timestep)

            flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                flow_pred_cond - flow_pred_uncond)
            temp_x0 = sample_scheduler.step(
                flow_pred.unsqueeze(0),
                t,
                latents.unsqueeze(0),
                return_dict=False)[0]
            latents = temp_x0.squeeze(0)

            
            # flow_pred_cond, x0_pred_cond = self.generator(latent_model_input, conditional_dict, timestep)
            # flow_pred_uncond, x0_pred_uncond = self.generator(latent_model_input, unconditional_dict, timestep)

            # x0_pred = x0_pred_uncond + self.args.guidance_scale * (
            #     x0_pred_cond - x0_pred_uncond)

            # temp_x0 = sample_scheduler.step(
            #     x0_pred.unsqueeze(0),
            #     t,
            #     latents.unsqueeze(0),
            #     return_dict=False)[0]
            # latents = temp_x0.squeeze(0)
            # self.latent_history.append(latents.cpu().clone())

    
        x0 = latents
        video = self.vae.decode_to_pixel(x0)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        del sample_scheduler

        if return_latents:
            return video, latents
        else:
            return video
        
    def _initialize_sample_scheduler(self, noise):
        if self.sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                self.sampling_steps, device=noise.device, shift=self.shift)
            self.timesteps = sample_scheduler.timesteps
        elif self.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(self.sampling_steps, self.shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=noise.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler



    def forward_noise_0_to_1000(self, x0: torch.Tensor, noise: torch.Tensor, save_every: int = 20):
        """
        x0: [B, T, C, H, W] (WAN latent layout used in this repo)
        Returns: dict {t: x_t} for t in [0..1000] (optionally subsampled by save_every)
        """
        # import ipdb;ipdb.set_trace()
        assert x0.ndim == 5, f"expected [B,T,C,H,W], got {tuple(x0.shape)}"
        device, dtype = x0.device, x0.dtype
        B, T, C, H, W = x0.shape

        sample_scheduler = self._initialize_sample_scheduler(x0)
        
        
        # In the repo, scheduler.set_timesteps(1000, training=True) is called in __init__. :contentReference[oaicite:3]{index=3}

        # Use ONE fixed eps so the trajectory is consistent (common for visualizing forward diffusion).
        # eps = torch.randn_like(x0)

        x0_flat = x0.flatten(0, 1)      # [B*T, C, H, W]
        noise_flat = noise.flatten(0, 1)    # [B*T, C, H, W]


        sample_timesteps = sample_scheduler.timesteps.cpu().tolist()
        
        # import ipdb;ipdb.set_trace()
        out = {}
        # for t in range(0, 1001, save_every):
        for t in sample_timesteps[::-1]:
            tvec = torch.full((B * T,), t, device=device, dtype=torch.long)
            
            xt_flat = sample_scheduler.add_noise(x0_flat, noise_flat, tvec)   # [B*T, C, H, W] :contentReference[oaicite:4]{index=4}
            out[t] = xt_flat.unflatten(0, (B, T)).to(dtype)          # [B, T, C, H, W]

        return out


    @torch.no_grad()
    def renoise_latents(
        self,
        x0: torch.Tensor,
        text_prompts: List[str],
        return_latents=False,
    ) -> torch.Tensor:
        # Encode prompts (same as inference)
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )

        latents = x0

        # Same scheduler + same set_timesteps
        sample_scheduler = self._initialize_sample_scheduler(latents)

        # IMPORTANT: flip direction -> forward-time integration (x0 -> noise)
        # Your inference loop assumes sample_scheduler.timesteps is in reverse order (high->low).
        # We invert it here.
        timesteps_fwd = torch.flip(sample_scheduler.timesteps, dims=[0])

        for t in tqdm(timesteps_fwd):
            timestep = t * torch.ones([latents.shape[0], 21], device=latents.device, dtype=torch.float32)

            flow_pred_cond, _ = self.generator(latents, conditional_dict, timestep)
            flow_pred_uncond, _ = self.generator(latents, unconditional_dict, timestep)

            flow_pred = flow_pred_uncond + self.args.guidance_scale * (flow_pred_cond - flow_pred_uncond)

            try:
                latents = sample_scheduler.step(
                    flow_pred.unsqueeze(0),
                    t,
                    latents.unsqueeze(0),
                    return_dict=False
                )[0].squeeze(0)
            except Exception as e:
                print("here I am", e)
                import ipdb;ipdb.set_trace()

        del sample_scheduler

        if return_latents:
            return latents
        else:
            # returning the "renoised" latents; decoding makes no sense here
            return latents
from pipeline import SelfForcingTrainingPipeline
from pipeline.progressive_self_forcing_training import ProgressiveSelfForcingTrainingPipeline
import torch.nn.functional as F
from typing import Optional, Tuple
import torch

from model.base import SelfForcingModel




"""
11/12/2025: added the first_window dmd loss logic for first_window_dmd_loss
19/12/2025: adding regression to it, and cleaning things up

13/01/2026: changing the method to perform latent form LPIPS from pixel form. 
"""


class ProgressiveDMD(SelfForcingModel):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: ProgressiveSelfForcingTrainingPipeline = None
        # self.inference_pipeline: SelfForcingTrainingPipeline = None

        # Step 2: Initialize all dmd hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None



        self.using_first_window_loss = getattr(args, "using_first_window_loss", False)
        self.use_dmd_regression_loss =  getattr(args, "use_dmd_regression_loss", False)
        self.use_first_window_dmd_regression_loss =  getattr(args, "use_first_window_dmd_regression_loss", False)
        
        
        self.first_window_loss_scale_scheduler = getattr(args, "first_window_loss_scale_scheduler", 'use_1')
        
        self.reg_loss_method = getattr(args, "reg_loss_method", 'lpips')
        self.reg_loss_coefficient =  getattr(args, "reg_loss_coefficient", 1.0)
        
        self.device = device


        
        
        # need to load in LPIPS if doing regression loss
        if self.reg_loss_method=='lpips':
            # lpips needs to decode to video into pixel form
            from elatentlpips import ELatentLPIPS
            self.loss_fn_vgg = ELatentLPIPS(encoder="flux", augment='bg', eval_mode=True).to(dtype=torch.float32, device=device)
            # sd3 leads to pure noise versus pure noise of 1.8211, which is hard to interprete. Flux leads
            # to 0.49955, which should be more expected
            # self.loss_fn_vgg = ELatentLPIPS(encoder="flux", augment='bg', eval_mode=True).to(dtype=torch.float32, device=device)
            for p in self.loss_fn_vgg.parameters():
                p.requires_grad_(False)
            self.loss_fn_vgg.requires_grad_(False)
            
            
            

    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Uses the fake and real score models to denoise, and returns the grad diff.


        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Compute the fake score
        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        if self.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=timestep
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep
        )

        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = (pred_fake_image - pred_real_image)

        # TODO: Change the normalizer for causal teacher
        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            p_real = (estimated_clean_image_or_video - pred_real_image)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach()
        }

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculates DMD loss by calling _compute_kl_grad() on a stochastically noise timestep for the noise
        

        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video

        # john: see if the Nan loss comes from the latent itself
        assert not torch.isnan(original_latent.double()).any().item(), "Error: original_latent has Nan"

        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )

            # TODO:should we change it to `timestep = self.scheduler.timesteps[timestep]`?
            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

            # Step 2: Compute the KL grad
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict
            )

            
        # john: see if the Nan loss comes from the latent itself
        assert not torch.isnan((original_latent.double() - grad.double()).detach()).any().item(), "Error: original_latent-grad has Nan"
        
        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")


        first_window_loss_scale = self.first_window_loss_scale()
        # add in dmd first chunk loss. only perform this if doing dmd loss
        if self.using_first_window_loss:
    
            first_window_dmd_loss = 0.5 * F.mse_loss(original_latent.double()[self.first_window_mask], 
                                        (original_latent.double() - grad.double()).detach()[self.first_window_mask], 
                                        reduction="mean")
                    
            first_window_log_dict = {
                    "first_window_dmdtrain_gradient_norm": torch.mean(torch.abs(grad[self.first_window_mask].detach()))
            }
        else:
            first_window_loss = 0
            first_window_log_dict = {}
                        
                
        dmd_loss_info = {
            "dmd_loss": dmd_loss,
            "first_window_dmd_loss": first_window_dmd_loss,
            "first_window_loss_scale": first_window_loss_scale
        }
        
        dmd_log_dict.update(first_window_log_dict)
        return dmd_loss_info, dmd_log_dict
                        

    
    def set_first_window_mask(self, noise_shape):
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()
        first_window_gradient_mask = torch.ones(noise_shape, dtype=torch.bool)
        chunks_for_first_window = self.inference_pipeline.first_window_size * self.inference_pipeline.num_frame_per_block
        first_window_gradient_mask[:, chunks_for_first_window:] = False
        
        self.first_window_mask = first_window_gradient_mask
        self.chunks_for_first_window = chunks_for_first_window 
    
    def calculate_lpips_loss(self, 
                            cleaned_video_latents: torch.tensor, 
                            noise_latents: torch.tensor,
                            conditional_dict: dict
                            ):
        """
        Applies lpips regression loss similar to the one inside of DMD.
        Returns a torch stack of the per-frame LPIPS values
        """
        
        noise_latents = noise_latents.squeeze(0).to(dtype=torch.float32, device=self.device).detach()
        cleaned_video_latents = cleaned_video_latents.squeeze(0).to(dtype=torch.float32, device=self.device)
        
        
        if not self.use_dmd_regression_loss or not self.using_first_window_loss:
            with torch.no_grad():
                generated_video_latents = self.inference_pipeline.inference(
                    noise=noise_latents,
                    **conditional_dict,
                )
        else:
            generated_video_latents = self.inference_pipeline.inference(
                    noise=noise_latents,
                    **conditional_dict,
                )

        
        # video shape [1, 81, 3, 480, 832]
        with torch.no_grad():

            assert generated_video_latents.shape == cleaned_video_latents.shape, f"Error: generated_video_latents.shape {generated_video_latents.shape} != cleaned_video_latents.shape {cleaned_video_latents.shape}"

        if not self.use_dmd_regression_loss or not self.using_first_window_loss:
            with torch.no_grad():
                
                lpips_values = [ self.loss_fn_vgg(generated_video_latents[:,idx], cleaned_video_latents[:,idx], normalize=True).mean() for idx in range(generated_video_latents.shape[1])]
        else:
            lpips_values = [ self.loss_fn_vgg(generated_video_latents[:,idx], cleaned_video_latents[:,idx], normalize=True).mean() for idx in range(generated_video_latents.shape[1])]
        
        print("==============================================lpips_values==============================================")
        
        print("lpips_values:", lpips_values)
        
        return torch.stack(lpips_values).squeeze() # makes this size [num_of_frames] from [num_of_frames,1,1,1]
        
    
    def first_window_loss_scale(self):
        """
        Applies a gradient mask over the loss formula for the first window.

        Implements some method so that, ideally, when self.pipeline.first_window_size=21 (the max), it is the same as
        just normal DMD, and only takes affect when first_window_size shrinks.

        """

        max_first_window_window_size = self.inference_pipeline.num_max_frames # get the # of latent frames from latent
        current_first_window_window_size = self.inference_pipeline.first_window_size * self.inference_pipeline.num_frame_per_block
        
        match self.first_window_loss_scale_scheduler:
            case 'normalized_inverted_first_window_size':
                loss_coeffcient_factor = (max_first_window_window_size - current_first_window_window_size) / max_first_window_window_size
            case 'use_0':
                loss_coeffcient_factor = 0
            case 'use_1':
                loss_coeffcient_factor = 1        
            case 'use_0_and_1':
                loss_coeffcient_factor = 0 if max_first_window_window_size==current_first_window_window_size else 1
            case _:
                loss_coeffcient_factor = 0
                
        return loss_coeffcient_factor



    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        regression_info: dict,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - regression_info should contain information about the regression loss:
                - cleaned_video_latents: tensor representing latent values for a video
                - noise_latent: the corresponding noise from which the cleaned_video_latents was generated from
                - reg_conditioning_signal: conditioing signal for generation
                
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        
        
        self.set_first_window_mask(image_or_video_shape)
        
        # Step 1: Unroll generator to obtain fake videos
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent
        )

        # Step 2: Compute the DMD loss
        dmd_loss_info, dmd_log_dict  = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to
        )

        # calculate regression loss
        # if self.use_dmd_regression_loss or self.using_first_window_loss and regression_info:
        if regression_info:
            reg_stack = self.calculate_lpips_loss(
                            regression_info["cleaned_video_latents"], 
                            regression_info["noise_latents"],
                            regression_info["reg_conditional_dict"])

            dmd_loss_info['dmd_reg_loss'] = torch.mean(reg_stack, dim=0)
            dmd_loss_info['first_window_dmd_reg_loss'] = torch.mean(reg_stack[:self.chunks_for_first_window], dim=0)
        else:
            dmd_loss_info['dmd_reg_loss'] = torch.mean([0])
            dmd_loss_info['first_window_dmd_reg_loss'] = torch.mean([0])
            
            
            
        # calculate the loss        
        dmd_losses = dmd_loss_info['dmd_loss'] + self.reg_loss_coefficient * dmd_loss_info['dmd_reg_loss']
        first_window_losses = dmd_loss_info['first_window_dmd_loss'] + self.reg_loss_coefficient * dmd_loss_info['first_window_dmd_reg_loss']
        first_window_loss_scale = dmd_loss_info["first_window_loss_scale"]
        
        dmd_loss_info['generator_loss'] = (dmd_losses + first_window_loss_scale * first_window_losses)

        print("==== dmd_loss_info ====")
        print(dmd_loss_info)
        print("==== done dmd_loss_info ====")
        
        dmd_log_dict.update({"first_window_size": self.inference_pipeline.first_window_size})
        
        return dmd_loss_info, dmd_log_dict

    
    
    
    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        # Step 1: Run generator on backward simulated noisy input
        with torch.no_grad():
            generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent
            )

        # Step 2: Compute the fake prediction
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=True
        )

        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep
        )

        # Step 3: Compute the denoising loss for the fake critic
        if self.args.denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )

        # Step 5: Debugging Log
        critic_log_dict = {
            "critic_timestep": critic_timestep.detach()
        }

        return denoising_loss, critic_log_dict
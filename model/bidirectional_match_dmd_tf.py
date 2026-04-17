from pipeline import SelfForcingTrainingPipeline
from pipeline import ProgressiveSelfForcingTrainingPipeline

import torch.nn.functional as F
from typing import Optional, Tuple
import torch

from model.base import SelfForcingModel


class BidirectionalMatchDMDTeacherForcing(SelfForcingModel):
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

        
        self.enable_latent_grad_diff  = getattr(args, "enable_latent_grad_diff", False)
        self.enable_latent_diff  = getattr(args, "enable_latent_diff", False)
        self.latent_diff_coef = getattr(args, "latent_diff_coef", 0)
        

        self.bidirectional_match_loss_weight = getattr(args, "bidirectional_match_loss_weight", 0)
        print("self.bidirectional_match_loss_weight", self.bidirectional_match_loss_weight)


        self.use_prior_sampled_noise_for_bidirecitonal_generation = getattr(args, "use_prior_sampled_noise_for_bidirecitonal_generation", False)
        print("self.use_prior_sampled_noise_for_bidirecitonal_generation", self.use_prior_sampled_noise_for_bidirecitonal_generation)
        self.save_noise = getattr(args, "save_noise", False)


        self.teacher_forcing = getattr(args, "teacher_forcing", False)
        if (self.teacher_forcing):
            print("Using teacher forcing as a loss")
        else:
            print("Using diffusion forcing as a loss")

        # have better data with using all 4 denoising steps
        self.perform_all_denoising_steps_for_bidirectional_generation = getattr(args, "perform_all_denoising_steps_for_bidirectional_generation", False)
        # for enabling teacher forcing, with flow loss instead of MSE on x0 pred
        self.tf_noised_latent = getattr(args, "tf_noised_latent", False)
        self.tf_perform_all_steps = getattr(args, "tf_perform_all_steps", False)


        self.scale_earlier_chunks = getattr(args, "scale_earlier_chunks", False)
        self.chunk_scale_min = getattr(args, "chunk_scale_min", 0.1)
        print("perform_all_denoising_steps_for_bidirectional_generation:", self.perform_all_denoising_steps_for_bidirectional_generation,
              "tf_noised_latent:", self.tf_noised_latent,
              "tf_perform_all_steps:", self.tf_perform_all_steps
        )




    def get_latent_boundary_diff(self,latent):
        """
        Returns the latent difference between start and ends of the latent chunk
        ie:
        temp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        temp[3::3] = [3, 6, 9, 12, 15, 18]
        temp[2:-1:3] = [2, 5, 8, 11, 14, 17]
        
        taking temp[3::3]-temp[2:-1:3] should yield gradients updates for temp[3::3], since we want to match the prior to the current generation
        """
        # assert self.inference_pipeline is not None, "Calling get_latent_boundary_diff() before intialized"
        # num_frames_per_chunk = self.inference_pipeline.num_frames_per_chunk
        num_frames_per_chunk = 3 # [][] hard coded 
        return latent[:, num_frames_per_chunk::num_frames_per_chunk, ...] - latent[:, num_frames_per_chunk-1:-1:num_frames_per_chunk, ...]

    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
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

        
        
        if self.enable_latent_grad_diff:
            
            pred_fake_image_diff = self.get_latent_boundary_diff(pred_fake_image)
            pred_real_image_diff = self.get_latent_boundary_diff(pred_real_image)
            grad_diff = (pred_fake_image_diff - pred_real_image_diff)
            
            if normalization:
                # Step 4: Gradient normalization (DMD paper eq. 8).
                estimated_clean_image_or_video_diff = self.get_latent_boundary_diff(estimated_clean_image_or_video)
                
                p_real_diff = (estimated_clean_image_or_video_diff - pred_real_image_diff)
                normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
                grad_diff = grad_diff / normalizer
            grad_diff = torch.nan_to_num(grad_diff)
            
            
            return grad, grad_diff, {
                "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
                "dmdtrain_gradient_diff_norm": torch.mean(torch.abs(grad_diff)).detach(),
                "timestep": timestep.detach()
            }

            
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

            if self.enable_latent_grad_diff:
                # Step 2: Compute the KL grad
                grad, grad_diff, dmd_log_dict = self._compute_kl_grad(
                    noisy_image_or_video=noisy_latent,
                    estimated_clean_image_or_video=original_latent,
                    timestep=timestep,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict
                )
            else:
                # Step 2: Compute the KL grad
                grad, dmd_log_dict = self._compute_kl_grad(
                    noisy_image_or_video=noisy_latent,
                    estimated_clean_image_or_video=original_latent,
                    timestep=timestep,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict
                )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
            
        # compute raw latent difference with DMD
        if self.enable_latent_grad_diff:
            original_latent_diff = self.get_latent_boundary_diff(original_latent)
            if gradient_mask is not None:
                
                dmd_diff_loss = 0.5 * F.mse_loss(original_latent_diff.double(
                )[gradient_mask], (original_latent_diff.double() - grad_diff.double()).detach()[gradient_mask], reduction="mean")
            else:
                dmd_diff_loss = 0.5 * F.mse_loss(original_latent_diff.double(
                ), (original_latent_diff.double() - grad_diff.double()).detach(), reduction="mean")
                
            
            dmd_loss = dmd_loss + self.latent_diff_coef * dmd_diff_loss
            
        # compute raw latent difference, no grad used for DMD
        if self.enable_latent_diff:
            original_latent_diff = self.get_latent_boundary_diff(original_latent)
            # takes zeros_like because diffence is already baked in
            dmd_loss = dmd_loss + self.latent_diff_coef * F.mse_loss(original_latent_diff, torch.zeros_like(original_latent_diff), reduction="mean")
            
        return dmd_loss, dmd_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
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
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        # if wanting to save noise, need to add it as a parameter after loading as a minimally invasive approach
        if self.save_noise:
            self._initialize_inference_pipeline()
            self.inference_pipeline.save_noise = self.save_noise
            
        sampled_noise = torch.randn(image_or_video_shape, device=self.device, dtype=self.dtype)
        # Step 1: Unroll generator to obtain fake videos
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
            sampled_noise=sampled_noise # give in a verison of the noise to the model
        )

        assert isinstance(self.inference_pipeline, ProgressiveSelfForcingTrainingPipeline), "Inference pipeline should be an instance of ProgressiveSelfForcingTrainingPipeline for bidirectional match loss"
        

        # Step 2: Compute the DMD loss
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to
        )


        print("Starting bidirectional match loss computation...")
        with torch.no_grad():
            
            prior_first_window_size = self.inference_pipeline.first_window_size
            self.inference_pipeline.first_window_size = 7 # do full bidirectional generation 

            # convert sampled noise into format for bidirecitonal inference
            if self.use_prior_sampled_noise_for_bidirecitonal_generation:
                sampled_noise_bidirecitonal_format = self.inference_pipeline.format_sampled_noise_for_bidirectional()
                self.inference_pipeline.sampled_noise = sampled_noise_bidirecitonal_format
            
            # when basing generation off of a bidirecitonal history, perform all denoising steps
            if self.perform_all_denoising_steps_for_bidirectional_generation: 
                self.inference_pipeline.prior_exit_flags = [len(self.inference_pipeline.denoising_step_list)-1 for _ in range(len(self.inference_pipeline.prior_exit_flags))]

            generated_video_latents_bidirectional, bidirectional_denoised_timestep_from, bidirectional_denoised_timestep_to = self.inference_pipeline.inference_with_trajectory(
                noise=sampled_noise,
                **conditional_dict,
                use_prior_exit_flag=True, # use the same # of denoising steps from before
                use_prior_sampled_noise=self.use_prior_sampled_noise_for_bidirecitonal_generation, # use the same randomly sampled noise from before
            )
            self.inference_pipeline.first_window_size = prior_first_window_size
            if self.save_noise:
                # restore the original noise after bidirectional generation
                self.inference_pipeline.sampled_noise = self.inference_pipeline.format_sampled_noise_for_causal()
            
        # perform teacher/diffusion forcing generation with bidirectional generation as the clean signal
        if self.teacher_forcing:
            # main idea: follow more closely to teacher forcing, and give the bidirecitonal_gen+noise as the noise used and ask it to perform an denoising step
            if self.tf_noised_latent:
                num_blocks = self.inference_pipeline.get_num_blocks(generated_video_latents_bidirectional)

                # noise to a timestep of 0 to 2, exclude the last timestep
                noised_timesteps = self.inference_pipeline.generate_and_sync_list(num_blocks, len(self.inference_pipeline.denoising_step_list)-1, 
                                                                        generated_video_latents_bidirectional.device)
                print("noised_timesteps:",noised_timesteps)
                # get timestep 
                noised_timesteps = (self.inference_pipeline.denoising_step_list[noised_timesteps] * torch.ones(
                        [generated_video_latents_bidirectional.shape[1] // self.num_frame_per_block], 
                        dtype=torch.long) ).to(device=generated_video_latents_bidirectional.device)
                noise = torch.randn_like(generated_video_latents_bidirectional)
            
                noisy_latent = self.scheduler.add_noise(
                    generated_video_latents_bidirectional.flatten(0, 1),
                    noise.flatten(0, 1),
                    torch.repeat_interleave(noised_timesteps, self.num_frame_per_block, dim=0)
                ).unflatten(0, generated_video_latents_bidirectional.shape[:2])

                # choice of doing all the same denoising steps or do the next timesteps
                if self.tf_perform_all_steps:
                    end_timestep = self.inference_pipeline.denoising_step_list[-1] * torch.ones_like(noised_timesteps)
                else:
                    end_timestep = noised_timesteps 

                tf_pred = self.inference_pipeline.inference_from_clean_noised(
                        noise=noisy_latent,
                        clean_history=generated_video_latents_bidirectional,
                        noised_timesteps=noised_timesteps.cpu(),
                        end_timestep=end_timestep.cpu(),
                        use_tf=True,
                        **conditional_dict
                ) 


                # loss = torch.nn.functional.mse_loss(flow_pred.float(), training_target.float())
                tf_loss = torch.nn.functional.mse_loss(
                    tf_pred.double(), generated_video_latents_bidirectional.double(), reduction='mean'
                )

                print("tf_loss", tf_loss.item())
                dmd_loss = dmd_loss + self.bidirectional_match_loss_weight * tf_loss
                dmd_log_dict.update({"tf_loss": torch.mean(tf_loss).detach()}) 

            else: # perform kv-cache replacement with MSE loss comparison over x0 predictions  
                print("perform generation with tf")
                tf_generation, denoised_timestep_from, denoised_timestep_to = self.inference_pipeline.inference_replace_history(
                    noise=sampled_noise,
                    clean_history=generated_video_latents_bidirectional.detach(),
                    use_tf=self.teacher_forcing,
                    **conditional_dict,
                    use_prior_exit_flag=True, # use the same # of denoising steps from before
                    use_prior_sampled_noise=self.use_prior_sampled_noise_for_bidirecitonal_generation, # use the same randomly sampled noise from before
                )

                if self.scale_earlier_chunks:
                    recreation_loss = F.mse_loss(tf_generation, generated_video_latents_bidirectional.detach(), 
                                                                   reduction="none").mean(dim=(2, 3, 4))
                    chunk_scaling = torch.repeat_interleave(
                        torch.linspace(1.0, self.chunk_scale_min, steps=num_blocks, device=recreation_loss.device),
                        self.num_frame_per_block, dim=0
                    )
                    recreation_loss = recreation_loss * chunk_scaling
                    recreation_loss = recreation_loss.mean()
                else:
                    recreation_loss = F.mse_loss(tf_generation, generated_video_latents_bidirectional.detach(), 
                                                                   reduction="mean")
                print("recreation_loss", recreation_loss.item())
                dmd_loss = dmd_loss + self.bidirectional_match_loss_weight * recreation_loss
                dmd_log_dict.update({"recreation_loss": torch.mean(recreation_loss).detach()}) 

        else:
            raise NotImplementedError("Diffusion forcing is not implemented yet, please set teacher_forcing to True to use teacher forcing")
    

        # for key in dmd_log_dict:
        #     if torch.is_tensor(dmd_log_dict[key]):
        #         dmd_log_dict[key] = dmd_log_dict[key].cpu()
        return dmd_loss, dmd_log_dict

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

"""
Same implementation as self-forcing/trainer/distillation.py, but with gradient accumulation added.  
Reference: https://github.com/NVlabs/LongLive/blob/main/trainer/distillation.py for changes regarding the grad accumulation.


Because the window for w1 changes, we don't want to polute the EMA generator with changes that are going be of a different 
window size, thus we must also update the EMA with the following options:

1) hard reset EMA on the current model weights (easiest), ideally requires that most of the window changes are done early on
2) 

"""

import gc
import logging

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import VideoRegressionShardingLMDBDataset

from utils.dataset import TextDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import CausVid, DMD, SiD, ProgressiveDMD
import torch
import wandb
import time
import os
import random

class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0
        
        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1) # john: add grad accumulation step

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)


        if self.is_main_process:
            print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")   
            print(f"Effective batch size: {config.batch_size * self.gradient_accumulation_steps * self.world_size}")

        # if self.is_main_process and not self.disable_wandb:
        #     wandb.login(host=config.wandb_host, key=config.wandb_key)
        #     wandb.init(
        #         config=OmegaConf.to_container(config, resolve=True),
        #         name=config.config_name,
        #         mode="online",
        #         entity=config.wandb_entity,
        #         project=config.wandb_project,
        #         dir=config.wandb_save_dir
        #     )
        
        if self.is_main_process: # always keep an instance of an offline wandb for training
            # wandb.login(host=config.wandb_host, key=config.wandb_key)
            
            os.makedirs(config.wandb_save_dir, exist_ok=True)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="offline",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )
            print(f"Wandb initalized. Saving to {config.wandb_save_dir}")
            
        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        elif config.distribution_loss == "progressive_dmd":
            self.model = ProgressiveDMD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)
        
        # Step 3a. Initalize the video dataloader
        reg_dataset = VideoRegressionShardingLMDBDataset(config.regression_data_path, max_pair=int(1e8))
        reg_sampler = torch.utils.data.distributed.DistributedSampler(
            reg_dataset, shuffle=True, drop_last=True)
        reg_dataloader = torch.utils.data.DataLoader(
            reg_dataset,
            batch_size=config.batch_size,
            sampler=reg_sampler,
            num_workers=8)
        

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
            print("REGRESSION DATASET SIZE %d" % len(reg_dataset))
            
        self.dataloader = cycle(dataloader)
        self.reg_dataloader = cycle(reg_dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None
        

    def resetEMA(self):
        """
        Copy over the data from the generator when performing the hard reset for the EMA parameters
        """
        for p, p_ema in zip(self.model.generator.parameters(), self.genereator_ema.parameters()):
            p_ema.data.copy_(p.data)

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))




    def fwdbwd_one_step(self, batch, train_generator, reg_batch=None):
        """ 
        John: same as self-forcing, but add in grad accumulation when performing loss.backwards(). Reported result is still without grad accumulation.

        Notabily generator_grad_norm and critic_grad_norm are no longer clipped here, but rather after accumulation in train().
        """
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        regression_info={}
        if reg_batch:
            if self.is_main_process:
                print("========= VIEW BATCHES =========")
                print(reg_batch["prompt"])
                print(batch["prompts"])
                print("========= DONE VIEW BATCHES =========")

            regression_info["reg_conditional_dict"] = self.model.text_encoder(
                text_prompts=reg_batch["prompt"])
            regression_info["noise_latents"] = reg_batch["noise"]
            regression_info["cleaned_video_latents"] = reg_batch["video"]
        
            
                
        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss_dict, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None,
                regression_info=regression_info
            )

            scaled_generator_loss = generator_loss_dict['generator_loss'] / self.gradient_accumulation_steps  # john: scale loss for grad accumulation
            scaled_generator_loss.backward()
            
            for k,v in generator_loss_dict.items():
                if torch.is_tensor(v):
                    generator_loss_dict[k] = v.detach()
            scaled_generator_loss = scaled_generator_loss.detach()
            
            
            # generator_grad_norm = self.model.generator.clip_grad_norm_(
            #     self.max_grad_norm_generator)

            # generator_log_dict.update({"generator_loss": generator_loss, # keep original
            #                            "generator_grad_norm": generator_grad_norm})

            generator_log_dict.update(generator_loss_dict)
            generator_log_dict.update({
                "generator_grad_norm": torch.tensor(0.0, device=self.device)
            }) # clip performed after accumulation
            
            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )
        scaled_critic_loss = critic_loss / self.gradient_accumulation_steps  # john: scale loss for grad accumulation
        scaled_critic_loss.backward()
        
        scaled_critic_loss = scaled_critic_loss.detach() # see if this does anything, likely not
        critic_loss = critic_loss.detach() # see if this does anything, likely not
        
        # critic_grad_norm = self.model.fake_score.clip_grad_norm_(
        #     self.max_grad_norm_critic)
        # critic_log_dict.update({"critic_loss": critic_loss, # log original loss
        #                         "critic_grad_norm": critic_grad_norm})
        critic_log_dict.update({"critic_loss": critic_loss, # log original loss
                                "critic_grad_norm": torch.tensor(0.0, device=self.device)}) # clip performed after accumulation
        return critic_log_dict

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )

        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    # John: Update pacing function      
    def update_pacing(self, generator_log_dict):
        self.pacing_has_updated = False
        # Update the stepping of the first window size according to the folllowing methods
        def none_wise(generator_log_dict):
            return 
        
        def step_wise(generator_log_dict):
            # base case
            # if self.w1_old == self.w1_new and self.w1_new == 1:
            #     return
            threshold_window_size = self.model.inference_pipeline.initial_first_window_size
            for i in range(len(self.config.pacing_kwargs.step_wise_thresholds)):
                threshold_step = self.config.pacing_kwargs.step_wise_thresholds[i][0]
                threshold_window_size = self.config.pacing_kwargs.step_wise_thresholds[i][1]
                if self.step< threshold_step:
                    break

            self.pacing_has_updated = self.w1_old > threshold_window_size
            # set w1_old and new to the same
            if self.pacing_has_updated:
                self.w1_old = self.w1_new
                self.w1_new = threshold_window_size
                
        def loss_wise(generator_log_dict):
            # base case
            if self.w1_old == self.w1_new and self.w1_new == 1:
                return 
            raise NotImplemented("Loss_wise pacing function to be implemented")


        match self.config.pacing_kwargs.pacing_function:
            case "none_wise":
                stepping_func = none_wise
            case "step_wise":
                stepping_func = step_wise
            case "loss_wise":
                stepping_func = loss_wise
            case _:
                stepping_func = none_wise

        def get_blending_method_probability(generator_log_dict):
            if self.config.pacing_kwargs.blending_method == 'none':
                self.w1_change_probability = 0 
            elif self.config.pacing_kwargs.blending_method == 'loss_based_stochastic':
                self.w1_change_probability = 0.5               
            else:
                raise NotImplemented("Loss_wise blending function to be implemented")
        
        stepping_func(generator_log_dict)
        get_blending_method_probability(generator_log_dict)
        
            
    
        self.model.inference_pipeline.first_window_size = random.choices([self.w1_old, self.w1_new], [1-self.w1_change_probability, self.w1_change_probability])[0]

        ## [][]TODO reset EMA to not leak old weights into newer verison
        # if self.pacing_has_updated:
        #     self.resetEMA()

                
    def train(self):
        start_step = self.step
        # reset_ema_next_step = False

        history_of_first_window_reg_loss = []
        stored_variances = []
        
        w1_change_probability = 0
        
        if self.model.inference_pipeline is None:
            self.model._initialize_inference_pipeline()
        self.w1_old = self.model.inference_pipeline.first_window_size
        self.w1_new = self.w1_old - 1
        
        while True:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)

            accumulated_generator_logs = []
            accumulated_critic_logs = []

            for accumulation_step in range(self.gradient_accumulation_steps):
                print(f"======== STEP: {self.step} On accumulation_step: {accumulation_step+1} ========")
                batch = next(self.dataloader)
                if TRAIN_GENERATOR:
                    reg_batch = next(self.reg_dataloader)
                    extra_gen = self.fwdbwd_one_step(batch, True, reg_batch=reg_batch)
                    print("done generator generation")
                    accumulated_generator_logs.append(extra_gen)
                    
                    for k,v in extra_gen.items():
                        if torch.is_tensor(v):
                            print(f"{k}: {v}, {v.device}, {v.numel()}, {v.requires_grad}")
                        else:
                            print(f"{k}: {v}")

                    if self.generator_ema is not None:
                        self.generator_ema.update(self.model.generator)
                    
                extra_crit = self.fwdbwd_one_step(batch, False)
                print("done critic generation")
                accumulated_critic_logs.append(extra_crit)

            # compute grad norm and update params
            if TRAIN_GENERATOR:
                generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
                generator_log_dict = merge_dict_list(accumulated_generator_logs)
                generator_log_dict["generator_grad_norm"] = generator_grad_norm

                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

                print("done generator step")
            else:
                generator_log_dict = {}

            # critic grad norm and update
            critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
            critic_log_dict = merge_dict_list(accumulated_critic_logs)
            critic_log_dict["critic_grad_norm"] = critic_grad_norm
            self.critic_optimizer.step()
            print("done critic step")

                                              
                
            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                print(f"Current step: {self.step+1}")
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    update_dict = {
                            k:(v.mean().item() if torch.is_tensor(v) else v) for k,v in generator_log_dict.items() 
                        }
                    # update_dict["first_window_size"] = self.model.inference_pipeline.first_window_size
                    
                    
                    wandb_loss_dict.update(
                        update_dict
                    )
                    
                    print(" ++++ VIEW just before saving ++++ ")
                    for key,value in update_dict.items():
                        print(f"{key}: {value}:")
                        if torch.is_tensor(value):
                            print(f"{key}: {value} {value.device}")
                        else:
                            print(f"{key}: {value}")
                                  
                        
                    
                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )
                
                # del critic_log_dict
                # del generator_log_dict
                

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)


                
            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            # update the pacing 
            self.update_pacing(generator_log_dict)
            
            ## probabilitically choose the w1 size according to the trend of reducing variance
#             if generator_log_dict and first_window_reg_loss := generator_log_dict['first_window_dmd_reg_loss'].mean().item() != 0:
#                 history_of_first_window_reg_loss.append(first_window_reg_loss)
                
#                 curr_variance = torch.var(torch.tensor(history_of_first_window_reg_loss[-4:]))
#                 stored_variances.append(curr_variance)
#                 if len(stored_variances)>=5 and torch.mean(torch.tensor(stored_variances[-5:-1])) < curr_variance:
#                     w1_change_probability += 0.2
                    
#                     if w1_change_probability > 1:
#                         w1_new = min(w1_new-1,1)
#                         w1_old = min(w1_old-1,1)
#                         stored_variances = [] # have at least 5 steps buffer that are dedicated to the new regimen
       
#                     # policy is to randomly set this value for every model
#                     import random 
#                     self.model.inference_pipeline.first_window_size = random.choices([w1_old, w1_new], [1-w1_change_probability, w1_change_probability])[0]
                
            ## stop training if loss is NaN
            # if "generator_loss" in generator_log_dict and "first_window_loss" in generator_log_dict:
            #     generator_loss_is_nan = generator_log_dict["generator_loss"].mean().item() is torch.nan
            #     first_window_loss_is_nan = generator_log_dict["first_window_loss"].mean().item() is torch.nan
            #     if generator_loss_is_nan or first_window_loss_is_nan:
            #         wandb.log(
            #             {"ERROR":f"Nan in loss. generator_loss_is_nan={generator_loss_is_nan}, first_window_loss_is_nan={first_window_loss_is_nan}"}, step=self.step)
                    # break

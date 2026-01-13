
21/11/2025
Goals:
This repo is to mainly view the affect of what can be called 'first-chunk priority' DMD. The hypothesis is that the autoregressive model needs a good first chunk generation in order to match the video made by the bidirectional teacher, so we focus early on getting the first chunk to match. 

Later on we incorporate causuallity through training, which I believe is an easier objective to learn.

There are not a lot to base this off of as there are not a lot of plots publically available for self-forcing/Wan/video diffusion model distillation. But we will try to do it here I guess.


03/12/2025
This is built upon the new_training_self-forcing/, where we were experimenting on having reinjection of latents.

This will be focued on getting actual implementation of a first-chunk focused training methods.

Idea 1: modification on DMD
Idea 2: modification on ODE initialization

Idea 2 is easier to implement, however I do not have the data for it. So we will try to runn Idea1 first.

But before all of this, we will need to modify configurations to actually run the Self-Forcing code on torchrun.

What is added:
System changes
* added vidprompt_dataset to prompts/
* config files to try to reproduce the Self-Forcing model, self_forcing_dmd_recreate.yaml and self_forcing_ode_recreate.yaml
* Focus is now on self_forcing_dmd_recreate.yaml
* get_paths.sh and make_env.sh which sets up the paths and environment respectively
* run_real_task.sh, which is an entrypoint for training using the Task

Actual code changes
* wan/wan_causal.py need to change CausalWan.set_gradient_checkpointing(value) because paramter enable=True is used when training. Have to have 2 parameters value and enable that default to False and do an or between them.


* because we want to run this as a task, we will need to source variables, so make source_paths.sh and then change utils/wan_wrapper.py FOLDER_PATH


9/12/2025
* updated make_env.sh and get_paths.sh
* added generate_ode_pairs.sh, to be ran on webstudio to get ODE pairs. Currently generating wan2.1-1.3B on guidance_step=3.0, timestep=5.0, on vidprom_filtered_extended.txt on webstudio with 2 A800s. More info in how_to_generate_ode_latents.sh
* updated scripts/generate_ode_pairs.py and scripts/create_lmdb_iterative.py to help faciliate these. Will likely need to update this.



branch: add in grad accumulation
* updates trainer/distllation.py to use gradient accumulation, taken from the implementation of LongLive
* updated get_paths.sh to accept arg2, which denotes the folder to run inside of arvd_repos/algorithm. Update run_real_task.sh to change based on if sourcing is already done to determine if we are in webstudio or a task


* also add in the progressive self-forcing inference code (ProgressiveCausalInferencePipeline), and verify that the generation is performing the correct chunk inside of videos/baseline2. This gives some confidence that ProgressiveSelfForcingInferencePipeline will work as it has the same first_window_size logic

* adding scripts/generate_videos_bidirectional.py and generate_videos_bidirectional.sh, which will generate videos from vidprom_filtered_extended.txt. However this method yields bad faces and movement. Thus I updated it to use the original wan codebase and just have it return the gaussian noise


15/12/2025:
branch: (progressive_naive) add progressive distillation methods, at least naiive ones:
* this updates a lot of the codebase in order to implement an idea of "large window for w1 early, progressively shrink w1 for causality later" namely:
   dwdawdwadwadwad * trainer/progressive_distillation.py
    * pipeline/progressive_self_forcing_training.py
    * model/progressive_dmd.py
* supporting changes across the board are required for the added method.
* this is built of of the 091225_add_grad_acc_temp repo, which is a pretty stable branch, as the trainning works, not sure about how well it works though.
* added some assert not torch.isnan() to check that there is no nan values generated for the latent or loss

* fix some stuff for video generation, as prior attempts were bad. `make_video_latents.sh` achieves this, other attempts were bad, albeit slowly on webstudio (10 minutes a video, even with flash_attn versus 3 minutes with standard wan that yielded worse output)
* add in a video dataset loader that will load in (noise, latent, prompt) triplets. 
    * added into utils/dataset.py as VideoRegressionShardingLMDBDataset
    * added in create_lmdb_shards_video.py to convert .pt files of the generated triplets into a lmdb dataset
    * updated utils/lmdb.py for versions that will handle video and noise
    * added a small test file test_lmdb_video_loading.py so we can quickly verify that it works interactively




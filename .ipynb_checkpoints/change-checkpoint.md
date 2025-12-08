
21/1/2025
Goals:
This repo is to mainly view the affect of what can be called 'first-chunk priority' DMD. The hypothesis is that the autoregressive model needs a good first chunk generation in order to match the video made by the bidirectional teacher, so we focus early on getting the first chunk to match. 

Later on we incorporate causuallity through training, which I believe is an easier objective to learn.

There are not a lot to base this off of as there are not a lot of plots publically available for self-forcing/Wan/video diffusion model distillation. But we will try to do it here I guess.


03/21/2025
This is built upon the new_training_self-forcing/, where we were experimenting on having reinjection of latents.

This will be focued on getting actual implementation of a first-chunk focused training methods.

Idea 1: modification on DMD
Idea 2: modification on ODE initialization

Idea 2 is easier to implement, however I do not have the data for it. So we will try to runn Idea1 first.

But before all of this, we will need to modify configurations to actually run the Self-Forcing code on torchrun.

What is added:
* added vidprompt_dataset to prompts/
* config files to try to reproduce the Self-Forcing model, self_forcing_dmd_recreate.yaml and self_forcing_ode_recreate.yaml
* Focus is now on self_forcing_dmd_recreate.yaml

* wan/wan_causal.py need to change CausalWan.set_gradient_checkpointing(value) because paramter enable=True is used when training. Have to have 2 parameters value and enable that default to False and do an or between them.

* because we want to run this as a task, we will need to source variables, so make source_paths.sh and then change utils/wan_wrapper.py FOLDER_PATH
# Update SelfForcingTrainingPipeline and ProgressiveSelfForcingTraining to utilize local_attn_size for caches.
* Currently training does utilize non-21 latent frame attention, but later on may want this to be a part of training
* I believe that the model (WanCausalModel) will internally only use the frames within local_attn_size, so this should be fine

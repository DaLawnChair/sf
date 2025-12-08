
# How was this done?
# After loading the pipeline, set initalize the caches, and then clear extra reserved with torch.cuda.empty_cache()
python3 emptied_cache_inference.py \
	--config_path configs/self_forcing_dmd.yaml \
	--output_folder videos/empty_cache \
	--checkpoint_path /home/ma-user/work/dataset/self_forcing/self_forcing/checkpoints/self_forcing_dmd.pt \
	--data_path prompts/twentyfive_MovieGenVideoBench_extended.txt \
	--use_ema


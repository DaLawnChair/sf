
python3 inference.py \
	--config_path configs/self_forcing_dmd.yaml \
	--output_folder videos/baseline2 \
	--checkpoint_path /home/ma-user/work/dataset/self_forcing/self_forcing/checkpoints/self_forcing_dmd.pt \
	--data_path prompts/twentyfive_MovieGenVideoBench_extended.txt \
	--use_ema


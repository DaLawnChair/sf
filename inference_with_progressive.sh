
python3 inference_with_progressive.py \
	--config_path configs/self_forcing_dmd.yaml \
	--output_folder videos/temp \
	--checkpoint_path /home/ma-user/work/dataset/self_forcing/self_forcing/checkpoints/self_forcing_dmd.pt \
	--data_path prompts/john_gorilla_banana.txt \
	--use_ema


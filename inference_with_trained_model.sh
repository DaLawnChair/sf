
#python3 inference_with_trained_model.py \
	#--config_path configs/self_forcing_dmd.yaml \
	#--output_folder videos/view_sf_recreate_ckpt_800 \
	#--checkpoint_path /home/ma-user/work/dataset/self_forcing/self_forcing/custom_dmd_training_grad_acc8/logs/checkpoint_model_000800/model.pt \
	#--data_path prompts/twentyfive_MovieGenVideoBench_extended.txt \
	#--use_ema

#python3 inference_with_trained_model.py \
        #--config_path configs/self_forcing_dmd.yaml \
        #--output_folder videos/view_sf_recreate_ckpt_600 \
        #--checkpoint_path /home/ma-user/work/dataset/self_forcing/self_forcing/custom_dmd_training_grad_acc8/logs/checkpoint_model_000600/model.pt \
        #--data_path prompts/twentyfive_MovieGenVideoBench_extended.txt \
        #--use_ema


CUDA_VISIBLE_DEVICES="0" python3 inference_with_trained_model.py \
        --config_path configs/self_forcing_dmd.yaml \
        --output_folder videos/view_sf_recreate_ckpt_100 \
        --checkpoint_path /home/ma-user/work/dataset/self_forcing/self_forcing/custom_dmd_training_grad_acc8/logs/checkpoint_model_000100/model.pt \
        --data_path prompts/twentyfive_MovieGenVideoBench_extended.txt \
        --use_ema &



CUDA_VISIBLE_DEVICES="1" python3 inference_with_trained_model.py \
        --config_path configs/self_forcing_dmd.yaml \
        --output_folder videos/view_sf_recreate_ckpt_60 \
        --checkpoint_path /home/ma-user/work/dataset/self_forcing/self_forcing/custom_dmd_training_grad_acc8/logs/checkpoint_model_000060/model.pt \
        --data_path prompts/twentyfive_MovieGenVideoBench_extended.txt \
        --use_ema &

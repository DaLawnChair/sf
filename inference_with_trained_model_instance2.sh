
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


#CUDA_VISIBLE_DEVICES="0" python3 inference_with_trained_model.py \
#        --config_path configs/self_forcing_dmd.yaml \
#        --output_folder videos/view_sf_recreate_ckpt_100 \
#        --checkpoint_path /home/ma-user/work/dataset/self_forcing/self_forcing/custom_dmd_training_grad_acc8/logs/checkpoint_model_000100/model.pt \
#        --data_path prompts/twentyfive_MovieGenVideoBench_extended.txt \
#        --use_ema &
#
#
#
#CUDA_VISIBLE_DEVICES="1" python3 inference_with_trained_model.py \
#        --config_path configs/self_forcing_dmd.yaml \
#        --output_folder videos/view_sf_recreate_ckpt_60 \
#        --checkpoint_path /home/ma-user/work/dataset/self_forcing/self_forcing/custom_dmd_training_grad_acc8/logs/checkpoint_model_000060/model.pt \
#        --data_path prompts/twentyfive_MovieGenVideoBench_extended.txt \
#        --use_ema &



# CUDA_VISIBLE_DEVICES="0" python3 inference_with_trained_model.py \
#         --config_path configs/150126_evaluate_baselines/elantentlpips_no_ode_init_pacing-step_wise_blend-none_webstudio.yaml \
# 	--output_folder videos/elatentlpips_run_basically_bidirecitonal_step_300 \
#         --checkpoint_path /home/ma-user/work/dataset/john_env/self_forcing_clone/self_forcing/elantentlpips_no_ode_init_pacing-step_wise_blend-none_webstudio_test/logs/checkpoint_model_000300/model.pt \
#         --data_path prompts/twentyfive_MovieGenVideoBench_extended.txt \
#         --use_ema &


#===================================================================================================
# experiments with stepwise progression on 200,250,300,350,400,450,500
# 200 step -> window size=7
# CUDA_VISIBLE_DEVICES="1" python3 inference_with_trained_model.py \
#         --config_path /home/ma-user/work/dataset/john_env/self_forcing_clone/self_forcing/model_training/test_naiive_stepwise_window_progression_params_match_data/elantentlpips_no_ode_init_pacing-step_wise_blend-none.yaml \
# 	--output_folder videos/test_naiive_stepwise_window_progression_params_match_data_000200 \
#         --checkpoint_path /home/ma-user/work/dataset/john_env/self_forcing_clone/self_forcing/model_training/test_naiive_stepwise_window_progression_params_match_data/logs/checkpoint_model_000200/model.pt \
#         --data_path prompts/five_MovieGenVideoBench_extended.txt \
#         --use_ema \
#         --first_window_size 7
        
        
# 200 step -> window size=7
# CUDA_VISIBLE_DEVICES="0" python3 inference_with_trained_model.py \
#         --config_path /home/ma-user/work/dataset/john_env/self_forcing_clone/self_forcing/model_training/test_naiive_stepwise_window_progression_params_match_data/elantentlpips_no_ode_init_pacing-step_wise_blend-none.yaml \
# 	--output_folder videos/test_naiive_stepwise_window_progression_params_match_data_000250 \
#         --checkpoint_path /home/ma-user/work/dataset/john_env/self_forcing_clone/self_forcing/model_training/test_naiive_stepwise_window_progression_params_match_data/logs/checkpoint_model_000250/model.pt \
#         --data_path prompts/five_MovieGenVideoBench_extended.txt \
#         --use_ema \
#         --first_window_size 6 &

# CUDA_VISIBLE_DEVICES="1" python3 inference_with_trained_model.py \
#         --config_path /home/ma-user/work/dataset/john_env/self_forcing_clone/self_forcing/model_training/test_naiive_stepwise_window_progression_params_match_data/elantentlpips_no_ode_init_pacing-step_wise_blend-none.yaml \
# 	--output_folder videos/test_naiive_stepwise_window_progression_params_match_data_000300 \
#         --checkpoint_path /home/ma-user/work/dataset/john_env/self_forcing_clone/self_forcing/model_training/test_naiive_stepwise_window_progression_params_match_data/logs/checkpoint_model_000300/model.pt \
#         --data_path prompts/five_MovieGenVideoBench_extended.txt \
#         --use_ema \
#         --first_window_size 5 &

# CUDA_VISIBLE_DEVICES="0" python3 inference_with_trained_model.py \
#         --config_path /home/ma-user/work/dataset/john_env/self_forcing_clone/self_forcing/model_training/test_naiive_stepwise_window_progression_params_match_data/elantentlpips_no_ode_init_pacing-step_wise_blend-none.yaml \
# 	--output_folder videos/test_naiive_stepwise_window_progression_params_match_data_000350 \
#         --checkpoint_path /home/ma-user/work/dataset/john_env/self_forcing_clone/self_forcing/model_training/test_naiive_stepwise_window_progression_params_match_data/logs/checkpoint_model_000350/model.pt \
#         --data_path prompts/five_MovieGenVideoBench_extended.txt \
#         --use_ema \
#         --first_window_size 4 &

# CUDA_VISIBLE_DEVICES="1" python3 inference_with_trained_model.py \
#         --config_path /home/ma-user/work/dataset/john_env/self_forcing_clone/self_forcing/model_training/test_naiive_stepwise_window_progression_params_match_data/elantentlpips_no_ode_init_pacing-step_wise_blend-none.yaml \
# 	--output_folder videos/test_naiive_stepwise_window_progression_params_match_data_000400 \
#         --checkpoint_path /home/ma-user/work/dataset/john_env/self_forcing_clone/self_forcing/model_training/test_naiive_stepwise_window_progression_params_match_data/logs/checkpoint_model_000400/model.pt \
#         --data_path prompts/five_MovieGenVideoBench_extended.txt \
#         --use_ema \
#         --first_window_size 3 &

# CUDA_VISIBLE_DEVICES="1" python3 inference_with_trained_model.py \
#         --config_path /home/ma-user/work/dataset/john_env/self_forcing_clone/self_forcing/model_training/test_naiive_stepwise_window_progression_params_match_data/elantentlpips_no_ode_init_pacing-step_wise_blend-none.yaml \
# 	--output_folder videos/test_naiive_stepwise_window_progression_params_match_data_000450 \
#         --checkpoint_path /home/ma-user/work/dataset/john_env/self_forcing_clone/self_forcing/model_training/test_naiive_stepwise_window_progression_params_match_data/logs/checkpoint_model_000450/model.pt \
#         --data_path prompts/five_MovieGenVideoBench_extended.txt \
#         --use_ema \
#         --first_window_size 2 &



# window_size=7
# for step in 000200 000250 000300 000350 000400 000450
# do
#     CUDA_VISIBLE_DEVICES="1" python3 inference_with_trained_model.py \
#         --config_path /home/ma-user/work/dataset/john_env/self_forcing_clone/self_forcing/model_training/test_naiive_stepwise_window_progression_params_match_data_with_init_with_dmd_reg_loss/elantentlpips_ode_init_pacing-step_wise_blend-none_with_dmd_reg_loss.yaml \
#         --output_folder videos/motion_prompts/test_naiive_stepwise_window_progression_params_match_data_with_init_with_dmd_reg_loss_$step \
#         --checkpoint_path /home/ma-user/work/dataset/john_env/self_forcing_clone/self_forcing/model_training/test_naiive_stepwise_window_progression_params_match_data_with_init_with_dmd_reg_loss/logs/checkpoint_model_$step/model.pt \
#         --data_path prompts/john_motion_prompts.txt \
#         --use_ema \
#         --first_window_size $window_size
#     ((window_size--))
# done

window_size=7
for step in 000200 000250 000300 000350 000400 000450
do
    CUDA_VISIBLE_DEVICES="1" python3 inference_with_trained_model.py \
        --config_path /home/ma-user/work/dataset/john_env/self_forcing_clone/self_forcing/model_training/test_bidirectional_capture_elatentlpips_params_match_data_with_init/elantentlpips_ode_init_pacing-none_wise_blend-none.yaml \
        --output_folder videos/motion_prompts/test_bidirectional_capture_elatentlpips_params_match_data_with_init_$step \
        --checkpoint_path /home/ma-user/work/dataset/john_env/self_forcing_clone/self_forcing/model_training/test_bidirectional_capture_elatentlpips_params_match_data_with_init/logs/checkpoint_model_$step/model.pt \
        --data_path prompts/john_motion_prompts.txt \
        --use_ema \
        --first_window_size $window_size
    # ((window_size--))
done
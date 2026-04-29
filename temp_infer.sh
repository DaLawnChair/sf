CUDA_VISIBLE_DEVICES="0" python long_video_test.py \
        --config_path /shared/john/model_training/john_env/self_forcing_clone/self_forcing/premade_models/self_forcing_dmd.yaml \
        --output_folder videos/temp \
        --checkpoint_path /shared/john/model_training/john_env/self_forcing_clone/self_forcing/premade_models/self_forcing_dmd.pt \
        --data_path prompts/john_cats.txt \
        --use_ema

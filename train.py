import argparse
import os
from omegaconf import OmegaConf
import wandb

from trainer import DiffusionTrainer, GANTrainer, ODETrainer, ScoreDistillationTrainer, ProgressiveScoreDistillationTrainer, ScoreDistillationCausalFakeScoreTrainer, ScoreDistillationNoFakeScoreModelTrainer

import torch 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize
    
    print("config.progressive_enabled",config.progressive_enabled)

    # get the filename of config_path
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb

    
    if config.trainer == "diffusion":
        trainer = DiffusionTrainer(config)
    elif config.trainer == "gan":
        trainer = GANTrainer(config)
    elif config.trainer == "ode":
        trainer = ODETrainer(config)
    elif config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
        
    elif config.trainer == "progressive_score_distillation":
        trainer = ProgressiveScoreDistillationTrainer(config)
    elif config.trainer == "score_distillation_causal_fake_score":
        trainer = ScoreDistillationCausalFakeScoreTrainer(config)
    elif config.trainer == "score_distillation_no_fake_score":
        trainer = ScoreDistillationNoFakeScoreModelTrainer(config)
    else:
        raise ValueError(f"config.trainer={config.trainer} not implemented") 
        
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()

Performed on 09/12/2025:


# Generation of Latents with `scripts/generate_ode_latents.py` 
Here we will use scripts/generate_ode_latents.py to generate (prompt, noise, latents) triplets, referred to as (noise,latent) pairs in the paper.

We note that there is an error with generate_ode_latents.py as variable prompt is a dict, and dicts cannot be used as keys when saving, so we fix this by saving the key instead:

```python
        torch.save(
            {prompt['prompts']: stored_data.cpu().detach()}, # john: prompt is a dict, use string
            os.path.join(args.output_folder, f"{prompt_index:05d}.pt")
        )
```

Also neeeded to add the folder directory to the system path in order to avoid import errors.

In order to run these, the following code was ran: (copied from `train_ode.sh`)
```bash
nproc_per_node=2
NNODES=1

export CUDA_VISIBLE_DEVICES="0,1"
torchrun --nnodes=$NNODES --nproc_per_node=$nproc_per_node --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR":"$MASTER_PORT \
  scripts/generate_ode_pairs.py \
  --caption_path "/home/ma-user/work/algorithm/arvd_repos/031225_progressive_sf/prompts/vidprom_filtered_extended.txt" \
  --output_folder "/home/ma-user/work/dataset/self_forcing/self_forcing/ode_init_guidance3.0_wan1.3B_vidprom/" \
  --timeshift_scale 5.0 \
  --guidance_scale 3.0 # noted inside of config/self_forcing_ode.yaml
```

The run takes about 28 hours to generate the triplets for 1000 samples, not sure how long it will take for the 250k samples for vidprom_filtered_extended, will likely just stop at one point.

## caveats
* The real score model used for self-forcing is the Wan2.1-14B model, and so the student and critic model is different from the teacher model for DMD loss. Not sure why. I generated the results using the Wan2.1-1.3B model
* The default guidance scale is 6.0 inside of the script, but they do not give an explicit value during training.


# Env
This was generated on 2 A800s on WebStudio using the john_env/self_forcing enviornment, which is a .whl enviornment save of the self_forcing/requirements.txt and some others, notably flash_attn==2.8.3, and it uses cuda128, torch2.8.0.


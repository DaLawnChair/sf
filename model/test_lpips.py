
import os
import torch
from elatentlpips import ELatentLPIPS
device = 'cuda'
loss_func = ELatentLPIPS(encoder="flux", augment='bg', eval_mode=True).to(dtype=torch.float32, device=device)

for p in loss_func.parameters():
    p.requires_grad_(False)
    
loss_func.requires_grad_(False)



# from utils.wan_wrapper import WanVAEWrapper


# vae = WanVAEWrapper().to('cuda')

generated_video_latents = torch.randn([1,21,16,60,104]).to('cuda')
cleaned_video_latents = torch.randn([1,21,16,60,104]).to('cuda')

lpips_values = [ loss_func(generated_video_latents[:,idx], cleaned_video_latents[:,idx], normalize=True).mean() for idx in range(generated_video_latents.shape[1])]


returned_value = torch.stack(lpips_values).squeeze()

reg_stack = torch.stack(lpips_values).squeeze()
real_loss = torch.mean(reg_stack, dim=0)

print("lpips_values", lpips_values)
print("returned_value", returned_value)
print("reg_stack", reg_stack)
print("real_loss", real_loss)



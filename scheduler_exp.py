import torch

num_inference_steps = 100 
num_train_timesteps = 1000
shift=3.0
sigma_min = 0.003 / 1.002
sigma_max = 1.0
inverse_timesteps=False
extra_one_step=False
reverse_sigmas=False

# set_timesteps()
denoising_stength = 1.0
training = True
sigma_start = sigma_min + (sigma_max-sigma_min) * denoising_stength
if extra_one_step:
	sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps+1)[:-1]
else:
	sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps)
if inverse_timesteps:
	sigmas = torch.flip(sigmas, dim=[0])
sigmas = shift * sigmas / (1+ (shift-1) * sigmas)
timesteps = sigmas * num_train_timesteps

if training:
	x = timesteps
	y = torch.exp(-2 * ((x - num_inference_steps / 2) /
                          num_inference_steps) ** 2)
	y_shifted = y - y.min()
	bsmntw_weighing = y_shifted * \
		(num_inference_steps / y_shifted.sum())
	linear_timesteps_weights = bsmntw_weighing
import ipdb;ipdb.set_trace()

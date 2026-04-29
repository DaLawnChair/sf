import torch
import torch.nn.functional as F
import random


class SelfForcing:
    def __init__(self):
        self.frames = torch.randn([1,21,16,60,104])

    def get_saturation(self, x1, saturation_ratio_min: float =0.7, saturation_ratio_max: float =2.0):
        if random.random() < 0.5:
            sat_factor = random.uniform(saturation_ratio_min, 1.0 - 1e-3)
        else:
            sat_factor = random.uniform(1.0 + 1e-3, saturation_ratio_max)
        latent_mean = torch.mean(x1, dim=1, keepdim=True)
        x1_saturated = (x1 - latent_mean) * sat_factor + latent_mean
        return x1_saturated
    
    def get_corrupt_noise_sigma(self, model_input, batch_size, corrupt_ratio=1 / 3, num_frames=None, is_frame_independent=False):
        if is_frame_independent:
            noise_sigma_shape = (batch_size, 1, num_frames)
        else:
            noise_sigma_shape = (batch_size,)
        noise_sigma = (
            torch.rand(size=noise_sigma_shape, device=model_input.device, dtype=model_input.dtype) * corrupt_ratio
        )
        while len(noise_sigma.shape) < model_input.ndim:
            noise_sigma = noise_sigma.unsqueeze(-1)
        return noise_sigma
    
    def same_frame_corruption(self, frame):
        return frame
    
    def downsample_corrupt(self, model_input, downsample_min_corrupt_ratio:float= 0.9, downsample_max_corrupt_ratio:float= 1.0):
        corrupt_ratio = random.uniform(downsample_min_corrupt_ratio, downsample_max_corrupt_ratio)

        is_5d = model_input.ndim == 5

        if is_5d:
            B, C, T, H, W = model_input.shape
            model_input = model_input.view(B * T, C, H, W)
        else:
            B, C, H, W = model_input.shape

        h0, w0 = model_input.shape[-2:]

        h1 = max(1, int(round(h0 * corrupt_ratio)))
        w1 = max(1, int(round(w0 * corrupt_ratio)))

        model_input = F.interpolate(model_input, size=(h1, w1), mode="bilinear", align_corners=False, antialias=True)

        model_input = F.interpolate(model_input, size=(h0, w0), mode="bilinear", align_corners=False, antialias=True)

        if is_5d:
            model_input = model_input.view(B, C, T, H, W)

        return model_input

    def corrupt_latent_history(self, history_start:int, history_end:int):
        """
        This function returns a corrupted version of the frames within bounds (history_start, history_end), where it will have equal changes to corrupt via
        saturaiton, noise, downsample+upsample, or no corruption. The corruption is applied frame-wise, meaning that for each frame in the history, 
        it will be independently decided which type of corruption to apply.

        Returns the corrupted history of the model from temporal indicies [history_start, ... ,history_end).
        Note: this assumes that the historical frames are stored inside as the same dimensionality as the frames, this will need to be modified within Self-Forcing code.

        """

        p_saturation = 0.25 
        p_noise = 0.25 
        p_same = 0.25
        p_downsample_upsample = 0.25 

        corrupt_history_storage = torch.zeros_like(self.frames)
        for idx in range(history_start, history_end):
            frame = self.frames[:, idx, ...]
            rand = torch.rand(1).item()
            if rand < p_saturation:
                # saturation corrupt
                frame = self.get_saturation(frame)
            elif rand < p_saturation + p_noise:
                # noise corruption
                noise_sigma = self.get_corrupt_noise_sigma(frame, batch_size=frame.shape[0], corrupt_ratio=1/3, num_frames=frame.shape[1], is_frame_independent=False)
                frame = (
                    noise_sigma * torch.randn_like(frame) + (1 - noise_sigma) * frame
                )
            elif rand < p_saturation + p_noise + p_downsample_upsample:
                # downsample and upsample corruption
                frame = self.downsample_corrupt(frame) # defaults to downsample_min_corrupt_ratio=0.9, downsample_max_corrupt_ratio=1.0
            else:
                # no corruption
                frame = self.same_frame_corruption(frame)

            # import ipdb;ipdb.set_trace()
            # corrupt_history_storage = torch.cat((corrupt_history_storage, frame.unsqueeze(1)), dim=1)
            corrupt_history_storage[:,idx,...] = frame
        return corrupt_history_storage 



if __name__ == "__main__":
    model = SelfForcing()

    resultant_history = SelfForcing.corrupt_latent_history(model, history_start=0, history_end=5)
    import ipdb;ipdb.set_trace()


    
        
         
        


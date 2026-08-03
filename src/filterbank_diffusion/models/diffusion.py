import torch
import torch.nn as nn
import numpy as np

class GaussianDiffusion(nn.Module):
    """
    Standard DDPM scheduler wrapping noise injection during training and 
    advanced sampling protocols utilizing Classifier-Free Guidance (CFG) and DDIM speedup.
    """
    def __init__(self, unet_model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = unet_model
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_prev = torch.cat([torch.ones(1), alphas[:-1]], dim=0)
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_prev) / (1.0 - alphas_bar))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("pred_noise_coef", betas / torch.sqrt(1.0 - alphas_bar))

    def q_sample(self, x_0, t, noise):
        sqrt_alpha_bar = self.sqrt_alphas_bar[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    @torch.no_grad()
    def p_sample_cfg(self, x_t, t, conditioning_C, class_labels, guidance_scale=3.0):
        batch_size = x_t.shape[0]
        null_labels = torch.full((batch_size,), fill_value=self.model.num_classes, dtype=torch.long, device=x_t.device)

        eps_conditional = self.model(x_t, t, conditioning_C, class_labels)
        eps_unconditional = self.model(x_t, t, conditioning_C, null_labels)
        eps_hat = eps_unconditional + guidance_scale * (eps_conditional - eps_unconditional)

        coef_recip = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        coef_noise = self.pred_noise_coef[t].view(-1, 1, 1, 1)
        mean = coef_recip * (x_t - coef_noise * eps_hat)
        
        if t[0] > 0:
            variance = self.posterior_variance[t].view(-1, 1, 1, 1)
            z = torch.randn_like(x_t)
            return mean + torch.sqrt(variance) * z
        else:
            return mean

    @torch.no_grad()
    def sample_loop_cfg(self, conditioning_C, class_labels, guidance_scale=3.0):
        """Standard 1000-step DDPM loop"""
        self.model.eval()
        device = conditioning_C.device
        batch_size, _, _, time_steps = conditioning_C.shape
        shape = (batch_size, 1, 64, time_steps)
        x = torch.randn(shape, device=device)
        
        for i in reversed(range(self.timesteps)):
            t_tensor = torch.full((batch_size,), fill_value=i, dtype=torch.long, device=device)
            x = self.p_sample_cfg(x, t_tensor, conditioning_C, class_labels, guidance_scale)
            
        return x

    @torch.no_grad()
    def sample_ddim_cfg(self, conditioning_C, class_labels, ddim_steps=25, guidance_scale=3.0, eta=0.0):
        """
        Fast DDIM Sampling Protocol (sub-sampling 1000 -> ddim_steps)
        Accelerates reconstruction by ~40x with deterministic trajectory.
        """
        self.model.eval()
        device = conditioning_C.device
        batch_size, _, _, time_steps = conditioning_C.shape
        shape = (batch_size, 1, 64, time_steps)
        
        # Select sub-sampled timesteps linearly across the 1000 schedule
        times = torch.linspace(0, self.timesteps - 1, ddim_steps + 1, device=device).long()
        time_pairs = list(zip(times[:-1], times[1:]))[::-1] # [(t_curr, t_prev), ...]
        
        x = torch.randn(shape, device=device)
        null_labels = torch.full((batch_size,), fill_value=self.model.num_classes, dtype=torch.long, device=device)

        for t_curr, t_prev in time_pairs:
            t_tensor = torch.full((batch_size,), fill_value=t_curr.item(), dtype=torch.long, device=device)
            
            # Predict noise with CFG
            eps_conditional = self.model(x, t_tensor, conditioning_C, class_labels)
            eps_unconditional = self.model(x, t_tensor, conditioning_C, null_labels)
            eps_hat = eps_unconditional + guidance_scale * (eps_conditional - eps_unconditional)
            
            alpha_bar_curr = self.alphas_bar[t_curr]
            alpha_bar_prev = self.alphas_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
            
            # Predict pristine x_0
            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_curr) * eps_hat) / torch.sqrt(alpha_bar_curr)
            
            # Compute DDIM direction
            sigma = eta * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_curr) * (1.0 - alpha_bar_curr / alpha_bar_prev))
            dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma**2, min=0.0)) * eps_hat
            
            noise = torch.randn_like(x) if sigma > 0 else 0.0
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma * noise

        return x

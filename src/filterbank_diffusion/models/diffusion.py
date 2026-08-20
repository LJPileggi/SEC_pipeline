import torch
import torch.nn as nn
import numpy as np

class GaussianDiffusion(nn.Module):
    """
    Standard DDPM scheduler wrapping noise injection during training and 
    advanced sampling protocols utilizing Classifier-Free Guidance (CFG) and DDIM speedup.
    Robustly patched against numerical underflow, division-by-zero, and NaN propagation.
    """
    def __init__(self, unet_model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = unet_model
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_prev = torch.cat([torch.ones(1), alphas[:-1]], dim=0)
        alphas_bar = torch.cumprod(alphas, dim=0)

        # Protection against zero division in variance calculation
        posterior_variance = betas * (1.0 - alphas_prev) / torch.clamp(1.0 - alphas_bar, min=1e-8)
        pred_noise_coef = betas / torch.sqrt(torch.clamp(1.0 - alphas_bar, min=1e-8))

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(torch.clamp(1.0 - alphas_bar, min=0.0)))
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(torch.clamp(1.0 / alphas, min=1e-8)))
        self.register_buffer("pred_noise_coef", pred_noise_coef)

    def q_sample(self, x_0, t, noise):
        sqrt_alpha_bar = self.sqrt_alphas_bar[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)
        
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return torch.nan_to_num(x_t, nan=0.0, posinf=20.0, neginf=-20.0)

    @torch.no_grad()
    def p_sample_cfg(self, x_t, t, conditioning_C, class_labels=None, guidance_scale=0.0):
        batch_size = x_t.shape[0]
        null_labels = torch.full((batch_size,), fill_value=self.model.num_classes, dtype=torch.long, device=x_t.device)

        # 🎯 MODULARITA: Se guidance_scale <= 0.0 o class_labels non e fornito, esegue 1 passata incondizionata
        if guidance_scale > 0.0 and class_labels is not None:
            eps_conditional = self.model(x_t, t, conditioning_C, class_labels)
            eps_unconditional = self.model(x_t, t, conditioning_C, null_labels)
            eps_hat = eps_unconditional + guidance_scale * (eps_conditional - eps_unconditional)
        else:
            target_labels = class_labels if class_labels is not None else null_labels
            eps_hat = self.model(x_t, t, conditioning_C, target_labels)

        eps_hat = torch.nan_to_num(eps_hat, nan=0.0, posinf=10.0, neginf=-10.0)

        coef_recip = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        coef_noise = self.pred_noise_coef[t].view(-1, 1, 1, 1)
        mean = coef_recip * (x_t - coef_noise * eps_hat)
        
        if t[0] > 0:
            variance = torch.clamp(self.posterior_variance[t].view(-1, 1, 1, 1), min=1e-8)
            z = torch.randn_like(x_t)
            x_next = mean + torch.sqrt(variance) * z
        else:
            x_next = mean

        x_next = torch.clamp(x_next, min=-20.0, max=20.0)
        return torch.nan_to_num(x_next, nan=0.0, posinf=20.0, neginf=-20.0)

    @torch.no_grad()
    def sample_loop_cfg(self, conditioning_C, class_labels=None, guidance_scale=0.0):
        """Standard 1000-step DDPM loop with numerical safety bounds"""
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
    def sample_ddim_cfg(self, conditioning_C, class_labels=None, ddim_steps=25, guidance_scale=0.0, eta=0.0):
        """
        Fast DDIM Sampling Protocol (sub-sampling 1000 -> ddim_steps)
        Supports both Conditional (CFG) and Fully Unconditional restoration.
        """
        self.model.eval()
        device = conditioning_C.device
        batch_size, _, _, time_steps = conditioning_C.shape
        shape = (batch_size, 1, 64, time_steps)
        
        times = torch.linspace(0, self.timesteps - 1, ddim_steps + 1, device=device).long()
        time_pairs = list(zip(times[:-1], times[1:]))[::-1]
        
        x = torch.randn(shape, device=device)
        null_labels = torch.full((batch_size,), fill_value=self.model.num_classes, dtype=torch.long, device=device)

        for t_curr, t_prev in time_pairs:
            t_tensor = torch.full((batch_size,), fill_value=t_curr.item(), dtype=torch.long, device=device)
            
            # 🎯 MODULARITA GUIDANCE
            if guidance_scale > 0.0 and class_labels is not None:
                eps_conditional = self.model(x, t_tensor, conditioning_C, class_labels)
                eps_unconditional = self.model(x, t_tensor, conditioning_C, null_labels)
                eps_hat = eps_unconditional + guidance_scale * (eps_conditional - eps_unconditional)
            else:
                target_labels = class_labels if class_labels is not None else null_labels
                eps_hat = self.model(x, t_tensor, conditioning_C, target_labels)
            
            eps_hat = torch.nan_to_num(eps_hat, nan=0.0, posinf=10.0, neginf=-10.0)
            
            alpha_bar_curr = self.alphas_bar[t_curr]
            alpha_bar_prev = self.alphas_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
            
            sqrt_alpha_curr = torch.sqrt(torch.clamp(alpha_bar_curr, min=1e-8))
            sqrt_one_minus_alpha = torch.sqrt(torch.clamp(1.0 - alpha_bar_curr, min=0.0))
            
            pred_x0 = (x - sqrt_one_minus_alpha * eps_hat) / sqrt_alpha_curr
            pred_x0 = torch.clamp(pred_x0, min=-20.0, max=20.0)
            
            denom = torch.clamp(1.0 - alpha_bar_curr, min=1e-8)
            sigma = eta * torch.sqrt(
                torch.clamp((1.0 - alpha_bar_prev) / denom * (1.0 - alpha_bar_curr / torch.clamp(alpha_bar_prev, min=1e-8)), min=0.0)
            )
            dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma**2, min=0.0)) * eps_hat
            
            noise = torch.randn_like(x) if sigma > 0 else 0.0
            x = torch.sqrt(torch.clamp(alpha_bar_prev, min=0.0)) * pred_x0 + dir_xt + sigma * noise
            
            x = torch.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0)

        return x

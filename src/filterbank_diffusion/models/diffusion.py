import torch
import torch.nn as nn
import numpy as np

class GaussianDiffusion(nn.Module):
    """
    Standard DDPM scheduler wrapping noise injection during training and 
    advanced sampling protocols utilizing Classifier-Free Guidance (CFG).
    """
    def __init__(self, unet_model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = unet_model
        self.timesteps = timesteps

        # Linear beta scheduling as required by standard generative frameworks
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_prev = torch.cat([torch.ones(1), alphas[:-1]], dim=0)
        alphas_bar = torch.cumprod(alphas, dim=0)

        # Pre-compute immutable coefficients mapped to buffers for hardware deployment
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))
        
        # Coefficients required for the standard analytical posterior reverse steps
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_prev) / (1.0 - alphas_bar))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("pred_noise_coef", betas / torch.sqrt(1.0 - alphas_bar))

    def q_sample(self, x_0, t, noise):
        """
        Forward analytical process: infects a pristine Mel tensor with gaussian noise at a given timestep t.
        Formula: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon
        """
        # Shapes unpacking matching batch boundaries
        sqrt_alpha_bar = self.sqrt_alphas_bar[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    @torch.no_grad()
    def p_sample_cfg(self, x_t, t, conditioning_C, class_labels, guidance_scale=3.0):
        """
        Reverse sampling step executing Classifier-Free Guidance (CFG).
        Evaluates the noise twice per step to push restoration profiles towards extreme clarity.
        """
        batch_size = x_t.shape[0]
        # Create a matching tensor filled entirely with the null token index (22 = "Unknown")
        null_labels = torch.full((batch_size,), fill_value=self.model.num_classes, dtype=torch.long, device=x_t.device)

        # 1. Evaluate conditional and unconditional noise prediction
        eps_conditional = self.model(x_t, t, conditioning_C, class_labels)
        eps_unconditional = self.model(x_t, t, conditioning_C, null_labels)

        # 2. Linear extrapolation guided by factor w
        # Formula: eps_hat = eps_uncond + w * (eps_cond - eps_uncond)
        eps_hat = eps_unconditional + guidance_scale * (eps_conditional - eps_unconditional)

        # 3. Standard reverse posterior execution to recover x_{t-1}
        coef_recip = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        coef_noise = self.pred_noise_coef[t].view(-1, 1, 1, 1)
        
        mean = coef_recip * (x_t - coef_noise * eps_hat)
        
        # Inject standard noise fallback if timestep is above the final convergence layer
        if t[0] > 0:
            variance = self.posterior_variance[t].view(-1, 1, 1, 1)
            z = torch.randn_like(x_t)
            return mean + torch.sqrt(variance) * z
        else:
            return mean

    @torch.no_grad()
    def sample_loop_cfg(self, conditioning_C, class_labels, guidance_scale=3.0):
        """
        Full inference loop: triggers 1000 iterative loops down to t=0 starting from pure VRAM noise.
        Pipelined directly in memory without causing storage bottlenecks.
        """
        self.model.eval()
        device = conditioning_C.device
        shape = conditioning_C.shape # Mapped directly from the [B, 1, 64, 700] mask anchor
        
        # Initialize latent grid with pure white gaussian noise
        x = torch.randn(shape, device=device)
        
        # Iterate backward through the entire pre-computed timeline
        for i in reversed(range(self.timesteps)):
            t_tensor = torch.full((shape[0],), fill_value=i, dtype=torch.long, device=device)
            x = self.p_sample_cfg(x, t_tensor, conditioning_C, class_labels, guidance_scale)
            
        return x # Returns pristine reconstructed Log-Mel tensor ready for HTS-AT injection

# diffusion.py
import torch
import torch.nn as nn

class ConditionalGaussianDiffusion(nn.Module):
    """
    Standard DDPM / DDIM scheduler for Image-to-Image Super-Resolution (Ho et al. Cascaded Diffusion).
    Diffuses pristine spectrogram x_0 and reconstructs it directly conditioned on x_cond.
    """
    def __init__(self, unet_model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = unet_model
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_prev = torch.cat([torch.ones(1), alphas[:-1]], dim=0)
        alphas_bar = torch.cumprod(alphas, dim=0)

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
        """Standard DDPM forward process: z_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*noise."""
        sqrt_alpha_bar = self.sqrt_alphas_bar[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return torch.nan_to_num(x_t, nan=0.0, posinf=20.0, neginf=-20.0)

    @torch.no_grad()
    def sample_ddim(self, x_cond, fraction_id=None, ddim_steps=25, eta=0.0):
        """
        Fast DDIM reverse sampling directly recovering x_0 pristine target.
        """
        self.model.eval()
        device = x_cond.device
        batch_size, _, freq_bins, time_steps = x_cond.shape
        shape = (batch_size, 1, freq_bins, time_steps)

        times = torch.linspace(0, self.timesteps - 1, ddim_steps + 1, device=device).long()
        time_pairs = list(zip(times[:-1], times[1:]))[::-1]

        # Start from standard normal noise in data space
        x_t = torch.randn(shape, device=device)

        for t_curr, t_prev in time_pairs:
            t_tensor = torch.full((batch_size,), fill_value=t_curr.item(), dtype=torch.long, device=device)
            eps_hat = self.model(x_t, t_tensor, x_cond, fraction_id=fraction_id)
            eps_hat = torch.nan_to_num(eps_hat, nan=0.0, posinf=10.0, neginf=-10.0)

            alpha_bar_curr = self.alphas_bar[t_curr]
            alpha_bar_prev = self.alphas_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

            sqrt_alpha_curr = torch.sqrt(torch.clamp(alpha_bar_curr, min=1e-8))
            sqrt_one_minus_alpha = torch.sqrt(torch.clamp(1.0 - alpha_bar_curr, min=0.0))

            pred_x0 = (x_t - sqrt_one_minus_alpha * eps_hat) / sqrt_alpha_curr
            pred_x0 = torch.clamp(pred_x0, min=-20.0, max=20.0)

            denom = torch.clamp(1.0 - alpha_bar_curr, min=1e-8)
            sigma = eta * torch.sqrt(
                torch.clamp((1.0 - alpha_bar_prev) / denom * (1.0 - alpha_bar_curr / torch.clamp(alpha_bar_prev, min=1e-8)), min=0.0)
            )
            dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma**2, min=0.0)) * eps_hat

            noise = torch.randn_like(x_t) if sigma > 0 else 0.0
            x_t = torch.sqrt(torch.clamp(alpha_bar_prev, min=0.0)) * pred_x0 + dir_xt + sigma * noise
            x_t = torch.nan_to_num(x_t, nan=0.0, posinf=20.0, neginf=-20.0)

        return torch.clamp(x_t, min=-20.0, max=20.0)

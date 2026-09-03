# diffusion.py
import torch
import torch.nn as nn

class GaussianDiffusionResidual(nn.Module):
    """
    DDPM / DDIM scheduler for Residual Diffusion.
    Models high-frequency spectral residual prediction (Delta X) and reconstructs
    pristine mel-spectrograms via deterministic additive base bypass: X_final = X_interp + Delta X.
    """
    def __init__(self, unet_model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = unet_model
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_prev = torch.cat([torch.ones(1), alphas[:-1]], dim=0)
        alphas_bar = torch.cumprod(alphas, dim=0)

        # Numerical bounds against division by zero
        posterior_variance = betas * (1.0 - alphas_prev) / torch.clamp(1.0 - alphas_bar, min=1e-8)
        pred_noise_coef = betas / torch.sqrt(torch.clamp(1.0 - alphas_bar, min=1e-8))

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(torch.clamp(1.0 - alphas_bar, min=0.0)))
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(torch.clamp(1.0 / alphas, min=1e-8)))
        self.register_buffer("pred_noise_coef", pred_noise_coef)

    def q_sample(self, delta_x_0, t, noise):
        """
        Forward diffusion process targeting residual spectrogram Delta X = X_pristine - X_interp.
        """
        sqrt_alpha_bar = self.sqrt_alphas_bar[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)
        x_t = sqrt_alpha_bar * delta_x_0 + sqrt_one_minus_alpha_bar * noise
        return torch.nan_to_num(x_t, nan=0.0, posinf=20.0, neginf=-20.0)

    @torch.no_grad()
    def sample_ddim(self, x_interp, ddim_steps=25, eta=0.0):
        """
        Accelerated DDIM sampling (1000 -> ddim_steps).
        Computes residual trajectory and yields the reconstructed spectrogram: X_interp + Delta X.
        """
        self.model.eval()
        device = x_interp.device
        batch_size, _, freq_bins, time_steps = x_interp.shape
        shape = (batch_size, 1, freq_bins, time_steps)

        times = torch.linspace(0, self.timesteps - 1, ddim_steps + 1, device=device).long()
        time_pairs = list(zip(times[:-1], times[1:]))[::-1]

        # Initialize from pure standard normal noise in residual space
        delta_t = torch.randn(shape, device=device)

        for t_curr, t_prev in time_pairs:
            t_tensor = torch.full((batch_size,), fill_value=t_curr.item(), dtype=torch.long, device=device)
            eps_hat = self.model(delta_t, t_tensor, x_interp)
            eps_hat = torch.nan_to_num(eps_hat, nan=0.0, posinf=10.0, neginf=-10.0)

            alpha_bar_curr = self.alphas_bar[t_curr]
            alpha_bar_prev = self.alphas_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

            sqrt_alpha_curr = torch.sqrt(torch.clamp(alpha_bar_curr, min=1e-8))
            sqrt_one_minus_alpha = torch.sqrt(torch.clamp(1.0 - alpha_bar_curr, min=0.0))

            pred_delta_0 = (delta_t - sqrt_one_minus_alpha * eps_hat) / sqrt_alpha_curr
            pred_delta_0 = torch.clamp(pred_delta_0, min=-20.0, max=20.0)

            denom = torch.clamp(1.0 - alpha_bar_curr, min=1e-8)
            sigma = eta * torch.sqrt(
                torch.clamp((1.0 - alpha_bar_prev) / denom * (1.0 - alpha_bar_curr / torch.clamp(alpha_bar_prev, min=1e-8)), min=0.0)
            )
            dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma**2, min=0.0)) * eps_hat

            noise = torch.randn_like(delta_t) if sigma > 0 else 0.0
            delta_t = torch.sqrt(torch.clamp(alpha_bar_prev, min=0.0)) * pred_delta_0 + dir_xt + sigma * noise
            delta_t = torch.nan_to_num(delta_t, nan=0.0, posinf=20.0, neginf=-20.0)

        # Global deterministic skip addition
        return torch.clamp(x_interp + delta_t, min=-20.0, max=20.0)

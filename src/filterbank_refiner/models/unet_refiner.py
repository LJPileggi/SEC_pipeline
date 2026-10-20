import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class RefinerResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(16, in_channels // 2), num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.norm2 = nn.GroupNorm(num_groups=min(16, out_channels // 2), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return self.shortcut(x) + h


class MelRefinerUNet(nn.Module):
    """
    Lightweight 2D U-Net per il raffinamento ad alta frequenza da Mel cross-domain.
    Input: [B, 2, 64, 1152] (Concat di x_t e x_cond).
    Output: [B, 1, 64, 1152] (Predizione della velocità del campo di flusso v_t).
    """
    def __init__(self, in_channels=2, out_channels=1, base_channels=32, channel_mult=(1, 2, 4)):
        super().__init__()
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder
        c1 = base_channels * channel_mult[0]  # 32
        c2 = base_channels * channel_mult[1]  # 64
        c3 = base_channels * channel_mult[2]  # 128

        self.enc1 = RefinerResBlock(base_channels, c1, time_dim)
        self.down1 = nn.Conv2d(c1, c1, kernel_size=4, stride=2, padding=1)  # -> [32, 576]

        self.enc2 = RefinerResBlock(c1, c2, time_dim)
        self.down2 = nn.Conv2d(c2, c2, kernel_size=4, stride=2, padding=1)  # -> [16, 288]

        self.enc3 = RefinerResBlock(c2, c3, time_dim)
        self.down3 = nn.Conv2d(c3, c3, kernel_size=4, stride=2, padding=1)  # -> [8, 144]

        # Bottleneck
        self.mid_block1 = RefinerResBlock(c3, c3, time_dim)
        self.mid_block2 = RefinerResBlock(c3, c3, time_dim)

        # Decoder
        self.up3 = nn.ConvTranspose2d(c3, c3, kernel_size=4, stride=2, padding=1)
        self.dec3 = RefinerResBlock(c3 + c3, c2, time_dim)

        self.up2 = nn.ConvTranspose2d(c2, c2, kernel_size=4, stride=2, padding=1)
        self.dec2 = RefinerResBlock(c2 + c2, c1, time_dim)

        self.up1 = nn.ConvTranspose2d(c1, c1, kernel_size=4, stride=2, padding=1)
        self.dec1 = RefinerResBlock(c1 + c1, base_channels, time_dim)

        self.out_norm = nn.GroupNorm(num_groups=min(16, base_channels // 2), num_channels=base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x_t, x_cond, t):
        t_emb = self.time_mlp(t)
        x_in = torch.cat([x_t, x_cond], dim=1)

        h0 = self.init_conv(x_in)

        # Down
        h1 = self.enc1(h0, t_emb)
        d1 = self.down1(h1)

        h2 = self.enc2(d1, t_emb)
        d2 = self.down2(h2)

        h3 = self.enc3(d2, t_emb)
        d3 = self.down3(h3)

        # Mid
        m = self.mid_block1(d3, t_emb)
        m = self.mid_block2(m, t_emb)

        # Up con skip connections
        u3 = self.up3(m)
        u3 = self.dec3(torch.cat([u3, h3], dim=1), t_emb)

        u2 = self.up2(u3)
        u2 = self.dec2(torch.cat([u2, h2], dim=1), t_emb)

        u1 = self.up1(u2)
        u1 = self.dec1(torch.cat([u1, h1], dim=1), t_emb)

        return self.out_conv(F.silu(self.out_norm(u1)))

    @torch.no_grad()
    def sample_refined(self, x_cond, num_steps=10):
        """
        Euler ODE sampling deterministico (Conditional Flow Matching).
        Traiettoria pulita da x_cond (t=0) a X_0 (t=1).
        """
        device = x_cond.device
        B = x_cond.shape[0]
        
        # ✅ Il punto di partenza a t=0 è ESATTAMENTE x_cond (NESSUN RUMORE)
        x = x_cond.clone()
        dt = 1.0 / float(num_steps)

        for i in range(num_steps):
            t_val = i / float(num_steps)
            t_tensor = torch.full((B,), t_val, device=device, dtype=x.dtype)
            v_pred = self.forward(x, x_cond, t_tensor)
            x = x + v_pred * dt

        return x

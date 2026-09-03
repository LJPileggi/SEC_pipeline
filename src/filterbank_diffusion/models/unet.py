# unet.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Standard Transformer-style sinusoidal positional encoding for diffusion timesteps.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation block to inject temporal embeddings
    into convolutional feature activations.
    """
    def __init__(self, embedding_dim, channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, channels * 2)
        )

    def forward(self, x, emb):
        emb_out = self.mlp(emb).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = emb_out.chunk(2, dim=1)
        return gamma * x + beta

class AsymmetricConvBlock(nn.Module):
    """
    Asymmetric convolutional block with a local residual skip connection.
    Uses 1x7 kernels across time and 7x1 kernels across frequency.
    """
    def __init__(self, in_channels, out_channels, emb_dim=None):
        super().__init__()
        self.conv_time = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 7), padding=(0, 3))
        self.conv_freq = nn.Conv2d(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0))
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.SiLU()
        self.film = FiLMBlock(emb_dim, out_channels) if emb_dim is not None else None
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, emb=None):
        h = self.conv_time(x)
        h = self.conv_freq(h)
        h = self.norm(h)
        h = self.act(h)
        if self.film is not None and emb is not None:
            h = self.film(h, emb)
        return h + self.res_conv(x)

class SpectrogramUNet(nn.Module):
    """
    Agnostic Multi-Level Residual U-Net designed for 64x700 spectrograms.
    Receives symmetric 2-channel inputs [x_t, x_interp] without class labels or attention blocks.
    Reaches bottleneck resolution at (2 x 21) across 512 channels (~43.2M parameters).
    """
    def __init__(self, base_channels=64, emb_dim=256):
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU()
        )

        c = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 8]

        # 2-channel input: [x_t, x_interp] concatenated along channel dimension
        self.inc = AsymmetricConvBlock(2, c[0], emb_dim)

        # Downsampling path
        # Level 1: 64x700 -> 32x350
        self.down_conv1 = nn.Conv2d(c[0], c[0], kernel_size=3, stride=(2, 2), padding=1)
        self.down1_block = AsymmetricConvBlock(c[0], c[1], emb_dim)

        # Level 2: 32x350 -> 16x175
        self.down_conv2 = nn.Conv2d(c[1], c[1], kernel_size=3, stride=(2, 2), padding=1)
        self.down2_block = AsymmetricConvBlock(c[1], c[2], emb_dim)

        # Level 3: 16x175 -> 8x87
        self.down_conv3 = nn.Conv2d(c[2], c[2], kernel_size=3, stride=(2, 2), padding=1)
        self.down3_block = AsymmetricConvBlock(c[2], c[3], emb_dim)

        # Level 4: 8x87 -> 4x43
        self.down_conv4 = nn.Conv2d(c[3], c[3], kernel_size=3, stride=(2, 2), padding=1)
        self.down4_block = AsymmetricConvBlock(c[3], c[4], emb_dim)

        # Level 5: 4x43 -> 2x21
        self.down_conv5 = nn.Conv2d(c[4], c[4], kernel_size=3, stride=(2, 2), padding=1)
        self.down5_block = AsymmetricConvBlock(c[4], c[4], emb_dim)

        # Bottleneck (2x21, purely convolutional)
        self.mid1 = AsymmetricConvBlock(c[4], c[4], emb_dim)
        self.mid2 = AsymmetricConvBlock(c[4], c[4], emb_dim)

        # Upsampling path with skip connections
        # Level 5
        self.up4 = nn.ConvTranspose2d(c[4], c[4], kernel_size=2, stride=2)
        self.up_block4 = AsymmetricConvBlock(c[4] * 2, c[4], emb_dim)

        # Level 4
        self.up3 = nn.ConvTranspose2d(c[4], c[3], kernel_size=2, stride=2)
        self.up_block3 = AsymmetricConvBlock(c[3] * 2, c[3], emb_dim)

        # Level 3
        self.up2 = nn.ConvTranspose2d(c[3], c[2], kernel_size=2, stride=2)
        self.up_block2 = AsymmetricConvBlock(c[2] * 2, c[2], emb_dim)

        # Level 2
        self.up1 = nn.ConvTranspose2d(c[2], c[1], kernel_size=2, stride=2)
        self.up_block1 = AsymmetricConvBlock(c[1] * 2, c[1], emb_dim)

        # Level 1
        self.up0 = nn.ConvTranspose2d(c[1], c[0], kernel_size=2, stride=2)
        self.up_block0 = AsymmetricConvBlock(c[0] * 2, c[0], emb_dim)

        # Output head: predicts single-channel residual noise
        self.outc = nn.Conv2d(c[0], 1, kernel_size=1)

    def forward(self, x_t, t, x_interp):
        """
        x_t: [B, 1, 64, 700]
        t: [B]
        x_interp: [B, 1, 64, 700]
        """
        x_in = torch.cat([x_t, x_interp], dim=1)
        t_emb = self.time_embedding(t)

        # Encoder
        h0 = self.inc(x_in, t_emb)
        h1 = self.down1_block(self.down_conv1(h0), t_emb)
        h2 = self.down2_block(self.down_conv2(h1), t_emb)
        h3 = self.down3_block(self.down_conv3(h2), t_emb)
        h4 = self.down4_block(self.down_conv4(h3), t_emb)
        h5 = self.down5_block(self.down_conv5(h4), t_emb)

        # Bottleneck
        h_mid = self.mid1(h5, t_emb)
        h_mid = self.mid2(h_mid, t_emb)

        # Decoder with dynamic dimension matching for odd pixel bounds
        u4 = self.up4(h_mid)
        if u4.shape[-2:] != h4.shape[-2:]:
            u4 = F.interpolate(u4, size=h4.shape[-2:], mode='bilinear', align_corners=False)
        h_up4 = self.up_block4(torch.cat([u4, h4], dim=1), t_emb)

        u3 = self.up3(h_up4)
        if u3.shape[-2:] != h3.shape[-2:]:
            u3 = F.interpolate(u3, size=h3.shape[-2:], mode='bilinear', align_corners=False)
        h_up3 = self.up_block3(torch.cat([u3, h3], dim=1), t_emb)

        u2 = self.up2(h_up3)
        if u2.shape[-2:] != h2.shape[-2:]:
            u2 = F.interpolate(u2, size=h2.shape[-2:], mode='bilinear', align_corners=False)
        h_up2 = self.up_block2(torch.cat([u2, h2], dim=1), t_emb)

        u1 = self.up1(h_up2)
        if u1.shape[-2:] != h1.shape[-2:]:
            u1 = F.interpolate(u1, size=h1.shape[-2:], mode='bilinear', align_corners=False)
        h_up1 = self.up_block1(torch.cat([u1, h1], dim=1), t_emb)

        u0 = self.up0(h_up1)
        if u0.shape[-2:] != h0.shape[-2:]:
            u0 = F.interpolate(u0, size=h0.shape[-2:], mode='bilinear', align_corners=False)
        h_up0 = self.up_block0(torch.cat([u0, h0], dim=1), t_emb)

        return self.outc(h_up0)

# unet.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    """Standard Transformer-style sinusoidal positional encoding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = x[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation block."""
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
    """Asymmetric conv block (1x7 time, 7x1 freq) with local residual skip."""
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

class MultiHeadSelfAttention2D(nn.Module):
    """Leggerissimo layer di Self-Attention spaziale per feature map compresse."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        head_dim = C // self.num_heads
        q = q.view(B, self.num_heads, head_dim, H * W).transpose(-1, -2)
        k = k.view(B, self.num_heads, head_dim, H * W).transpose(-1, -2)
        v = v.view(B, self.num_heads, head_dim, H * W).transpose(-1, -2)
        
        scale = 1.0 / (head_dim ** 0.5)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(-1, -2).contiguous().view(B, C, H, W)
        return x + self.proj(out)

class SpectrogramUNet(nn.Module):
    """
    Enhanced 5-level U-Net:
    - 3-channel input: [x_t, x_cond, x_self_cond] for Self-Conditioning
    - Bottleneck Multi-Head Self-Attention
    - Multi-scale condition injection into the decoder
    """
    def __init__(self, base_channels=64, emb_dim=256, cond_channels=16):
        super().__init__()
        self.cond_channels = cond_channels

        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU()
        )
        self.res_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU()
        )
        self.fused_embedding = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.SiLU()
        )

        c = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 8]
        # c = [64, 128, 256, 512, 512]

        # 3 canali di ingresso: [x_t, x_cond, x_self_cond]
        self.inc = AsymmetricConvBlock(3, c[0], emb_dim)

        # Downsampling path
        self.down_conv1 = nn.Conv2d(c[0], c[0], kernel_size=3, stride=(2, 2), padding=1)
        self.down1_block = AsymmetricConvBlock(c[0], c[1], emb_dim)

        self.down_conv2 = nn.Conv2d(c[1], c[1], kernel_size=3, stride=(2, 2), padding=1)
        self.down2_block = AsymmetricConvBlock(c[1], c[2], emb_dim)

        self.down_conv3 = nn.Conv2d(c[2], c[2], kernel_size=3, stride=(2, 2), padding=1)
        self.down3_block = AsymmetricConvBlock(c[2], c[3], emb_dim)

        self.down_conv4 = nn.Conv2d(c[3], c[3], kernel_size=3, stride=(2, 2), padding=1)
        self.down4_block = AsymmetricConvBlock(c[3], c[4], emb_dim)

        self.down_conv5 = nn.Conv2d(c[4], c[4], kernel_size=3, stride=(2, 2), padding=1)
        self.down5_block = AsymmetricConvBlock(c[4], c[4], emb_dim)

        # Bottleneck: mid1 + Self-Attention (72 token) + mid2
        self.mid1 = AsymmetricConvBlock(c[4], c[4], emb_dim)
        self.attn_mid = MultiHeadSelfAttention2D(c[4], num_heads=4)
        self.mid2 = AsymmetricConvBlock(c[4], c[4], emb_dim)

        # Proiezione feature di x_cond per re-iniezione gerarchica
        self.cond_proj = nn.Conv2d(1, cond_channels, kernel_size=3, padding=1)

        # Upsampling path: Skip Connections + feature di x_cond riscalata
        self.up4 = nn.ConvTranspose2d(c[4], c[4], kernel_size=2, stride=2)
        self.up_block4 = AsymmetricConvBlock(c[4] * 2 + cond_channels, c[4], emb_dim)

        self.up3 = nn.ConvTranspose2d(c[4], c[3], kernel_size=2, stride=2)
        self.up_block3 = AsymmetricConvBlock(c[3] * 2 + cond_channels, c[3], emb_dim)

        self.up2 = nn.ConvTranspose2d(c[3], c[2], kernel_size=2, stride=2)
        self.up_block2 = AsymmetricConvBlock(c[2] * 2 + cond_channels, c[2], emb_dim)

        self.up1 = nn.ConvTranspose2d(c[2], c[1], kernel_size=2, stride=2)
        self.up_block1 = AsymmetricConvBlock(c[1] * 2 + cond_channels, c[1], emb_dim)

        self.up0 = nn.ConvTranspose2d(c[1], c[0], kernel_size=2, stride=2)
        self.up_block0 = AsymmetricConvBlock(c[0] * 2 + cond_channels, c[0], emb_dim)

        self.outc = nn.Conv2d(c[0], 1, kernel_size=1)

    def forward(self, x_t, t, x_cond, fraction_id=None, x_self_cond=None):
        if x_self_cond is None:
            x_self_cond = torch.zeros_like(x_t)

        x_in = torch.cat([x_t, x_cond, x_self_cond], dim=1)
        t_emb = self.time_embedding(t)

        if fraction_id is not None:
            if not isinstance(fraction_id, torch.Tensor):
                fraction_id = torch.full((x_t.shape[0],), fill_value=float(fraction_id), device=x_t.device)
            elif fraction_id.ndim == 0:
                fraction_id = fraction_id.expand(x_t.shape[0])
            frac_scalar = torch.log2(fraction_id.float().clamp(min=1.0))
            r_emb = self.res_embedding(frac_scalar)
            fused_emb = self.fused_embedding(torch.cat([t_emb, r_emb], dim=-1))
        else:
            dummy_res = torch.zeros_like(t_emb)
            fused_emb = self.fused_embedding(torch.cat([t_emb, dummy_res], dim=-1))

        # Mappa multi-scala del condizionamento
        c_feat = self.cond_proj(x_cond)

        # Encoder
        h0 = self.inc(x_in, fused_emb)
        h1 = self.down1_block(self.down_conv1(h0), fused_emb)
        h2 = self.down2_block(self.down_conv2(h1), fused_emb)
        h3 = self.down3_block(self.down_conv3(h2), fused_emb)
        h4 = self.down4_block(self.down_conv4(h3), fused_emb)
        h5 = self.down5_block(self.down_conv5(h4), fused_emb)

        # Bottleneck con Self-Attention
        h_mid = self.mid1(h5, fused_emb)
        h_mid = self.attn_mid(h_mid)
        h_mid = self.mid2(h_mid, fused_emb)

        # Decoder con Skip Connections e Condizionamento Multi-Scala
        c4 = F.interpolate(c_feat, size=h4.shape[-2:], mode='bilinear', align_corners=False)
        u4 = self.up4(h_mid)
        if u4.shape[-2:] != h4.shape[-2:]:
            u4 = F.interpolate(u4, size=h4.shape[-2:], mode='bilinear', align_corners=False)
        h_up4 = self.up_block4(torch.cat([u4, h4, c4], dim=1), fused_emb)

        c3 = F.interpolate(c_feat, size=h3.shape[-2:], mode='bilinear', align_corners=False)
        u3 = self.up3(h_up4)
        if u3.shape[-2:] != h3.shape[-2:]:
            u3 = F.interpolate(u3, size=h3.shape[-2:], mode='bilinear', align_corners=False)
        h_up3 = self.up_block3(torch.cat([u3, h3, c3], dim=1), fused_emb)

        c2 = F.interpolate(c_feat, size=h2.shape[-2:], mode='bilinear', align_corners=False)
        u2 = self.up2(h_up3)
        if u2.shape[-2:] != h2.shape[-2:]:
            u2 = F.interpolate(u2, size=h2.shape[-2:], mode='bilinear', align_corners=False)
        h_up2 = self.up_block2(torch.cat([u2, h2, c2], dim=1), fused_emb)

        c1 = F.interpolate(c_feat, size=h1.shape[-2:], mode='bilinear', align_corners=False)
        u1 = self.up1(h_up2)
        if u1.shape[-2:] != h1.shape[-2:]:
            u1 = F.interpolate(u1, size=h1.shape[-2:], mode='bilinear', align_corners=False)
        h_up1 = self.up_block1(torch.cat([u1, h1, c1], dim=1), fused_emb)

        c0 = c_feat
        u0 = self.up0(h_up1)
        if u0.shape[-2:] != h0.shape[-2:]:
            u0 = F.interpolate(u0, size=h0.shape[-2:], mode='bilinear', align_corners=False)
        h_up0 = self.up_block0(torch.cat([u0, h0, c0], dim=1), fused_emb)

        return self.outc(h_up0)

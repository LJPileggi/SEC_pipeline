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
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

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
    """Layer di Self-Attention spaziale per il flusso audio compresso."""
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

class GuidedCrossAttention2D(nn.Module):
    """
    Bottleneck Cross-Attention: il ramo audio (Query) viene guidato e condizionato 
    dalla rappresentazione latente delle ottave (Key, Value).
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm_audio = nn.GroupNorm(8, channels)
        self.norm_cond = nn.GroupNorm(8, channels)
        
        self.to_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.to_k = nn.Conv2d(channels, channels, kernel_size=1)
        self.to_v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x_audio, x_cond):
        B, C, Ha, Wa = x_audio.shape
        _, _, Hc, Wc = x_cond.shape
        
        ha = self.norm_audio(x_audio)
        hc = self.norm_cond(x_cond)
        
        q = self.to_q(ha)
        k = self.to_k(hc)
        v = self.to_v(hc)
        
        head_dim = C // self.num_heads
        q = q.view(B, self.num_heads, head_dim, Ha * Wa).transpose(-1, -2) # [B, heads, Na, head_dim]
        k = k.view(B, self.num_heads, head_dim, Hc * Wc).transpose(-1, -2) # [B, heads, Nc, head_dim]
        v = v.view(B, self.num_heads, head_dim, Hc * Wc).transpose(-1, -2) # [B, heads, Nc, head_dim]
        
        scale = 1.0 / (head_dim ** 0.5)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
        out = torch.matmul(attn, v) # [B, heads, Na, head_dim]
        
        out = out.transpose(-1, -2).contiguous().view(B, C, Ha, Wa)
        return x_audio + self.proj(out)

class BottleneckDualAttention(nn.Module):
    """
    Doppio stadio attentivo al bottleneck:
    1. Self-Attention: coerenza e correlazione armonica interna del segnale audio rumoroso.
    2. Guided Cross-Attention: ancoraggio alle feature fisiche della griglia a frazioni d'ottava (320 bin).
    3. FFN: raffinamento non lineare residuo.
    """
    def __init__(self, channels=512, num_heads=4):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention2D(channels, num_heads=num_heads)
        self.cross_attn = GuidedCrossAttention2D(channels, num_heads=num_heads)
        self.norm = nn.GroupNorm(8, channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1)
        )

    def forward(self, x_audio, x_cond):
        h = self.self_attn(x_audio)
        h = self.cross_attn(h, x_cond)
        h = h + self.ffn(self.norm(h))
        return h

class SpectrogramUNet(nn.Module):
    """
    Dual-Stream Asymmetric U-Net con:
    - Ramo di Ingresso Audio a 2 canali: [x_t, x_self_cond] (64 x 1152)
    - Ramo di Condizionamento Iper-Risoluto (F_ref=320, 1152)
    - Bottleneck Dual-Attention (Self-Attention + Guided Cross-Attention + FFN)
    - Re-iniezione multi-scala del condizionamento nel Decoder
    """
    def __init__(self, base_channels=64, emb_dim=256, cond_channels=16):
        super().__init__()
        self.cond_channels = cond_channels

        # Time & Resolution Embeddings
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

        # ----------------------------------------------------------------------
        # 1. RAMO PRINCIPALE (AUDIO): [x_t, x_self_cond] -> 2 canali (64 x 1152)
        # ----------------------------------------------------------------------
        self.inc_audio = AsymmetricConvBlock(2, c[0], emb_dim)
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

        # ----------------------------------------------------------------------
        # 2. RAMO CONDIZIONAMENTO DEDICATO: x_cond [1, 320, 1152]
        # ----------------------------------------------------------------------
        self.inc_cond = AsymmetricConvBlock(1, cond_channels, emb_dim)
        self.down_cond1 = nn.Sequential(
            nn.Conv2d(cond_channels, cond_channels, kernel_size=3, stride=(2, 2), padding=1),
            AsymmetricConvBlock(cond_channels, cond_channels, emb_dim)
        )
        self.down_cond2 = nn.Sequential(
            nn.Conv2d(cond_channels, cond_channels, kernel_size=3, stride=(2, 2), padding=1),
            AsymmetricConvBlock(cond_channels, cond_channels, emb_dim)
        )
        self.down_cond3 = nn.Sequential(
            nn.Conv2d(cond_channels, cond_channels, kernel_size=3, stride=(2, 2), padding=1),
            AsymmetricConvBlock(cond_channels, cond_channels, emb_dim)
        )
        self.down_cond4 = nn.Sequential(
            nn.Conv2d(cond_channels, cond_channels, kernel_size=3, stride=(2, 2), padding=1),
            AsymmetricConvBlock(cond_channels, cond_channels, emb_dim)
        )
        self.down_cond5 = nn.Sequential(
            nn.Conv2d(cond_channels, cond_channels, kernel_size=3, stride=(2, 2), padding=1),
            AsymmetricConvBlock(cond_channels, c[4], emb_dim)
        )

        # ----------------------------------------------------------------------
        # 3. BOTTLENECK: DUAL ATTENTION + FiLM MODULATION
        # ----------------------------------------------------------------------
        self.mid1 = AsymmetricConvBlock(c[4], c[4], emb_dim)
        self.bottleneck_dual_attn = BottleneckDualAttention(channels=c[4], num_heads=4)
        self.mid2 = AsymmetricConvBlock(c[4], c[4], emb_dim)

        # ----------------------------------------------------------------------
        # 4. DECODER CON SKIP CONNECTIONS E FUSIONE CONDIZIONAMENTO
        # ----------------------------------------------------------------------
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

        x_audio_in = torch.cat([x_t, x_self_cond], dim=1) # [B, 2, 64, 1152]
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

        # --- Forward Ramo Condizionamento ---
        c0 = self.inc_cond(x_cond, fused_emb) # [B, cond_channels, 320, 1152]
        c1 = self.down_cond1[1](self.down_cond1[0](c0), fused_emb)
        c2 = self.down_cond2[1](self.down_cond2[0](c1), fused_emb)
        c3 = self.down_cond3[1](self.down_cond3[0](c2), fused_emb)
        c4 = self.down_cond4[1](self.down_cond4[0](c3), fused_emb)
        c5 = self.down_cond5[1](self.down_cond5[0](c4), fused_emb) # [B, c[4], 10, 36]

        # --- Forward Ramo Principale Audio ---
        h0 = self.inc_audio(x_audio_in, fused_emb) # [B, 64, 64, 1152]
        h1 = self.down1_block(self.down_conv1(h0), fused_emb)
        h2 = self.down2_block(self.down_conv2(h1), fused_emb)
        h3 = self.down3_block(self.down_conv3(h2), fused_emb)
        h4 = self.down4_block(self.down_conv4(h3), fused_emb)
        h5 = self.down5_block(self.down_conv5(h4), fused_emb) # [B, c[4], 2, 36]

        # --- Bottleneck con Doppia Attenzione (Self + Guided Cross) ---
        h_mid = self.mid1(h5, fused_emb)
        h_mid = self.bottleneck_dual_attn(h_mid, c5)
        h_mid = self.mid2(h_mid, fused_emb)

        # Helper per l'adattamento dimensionale delle skip del condizionamento
        def match_and_cat(u, h_skip, c_skip):
            if u.shape[-2:] != h_skip.shape[-2:]:
                u = F.interpolate(u, size=h_skip.shape[-2:], mode='bilinear', align_corners=False)
            if c_skip.shape[-2:] != h_skip.shape[-2:]:
                c_skip = F.interpolate(c_skip, size=h_skip.shape[-2:], mode='bilinear', align_corners=False)
            return torch.cat([u, h_skip, c_skip], dim=1)

        # --- Decoder con Skip Connections e Condizionamento Multi-Scala ---
        h_up4 = self.up_block4(match_and_cat(self.up4(h_mid), h4, c4), fused_emb)
        h_up3 = self.up_block3(match_and_cat(self.up3(h_up4), h3, c3), fused_emb)
        h_up2 = self.up_block2(match_and_cat(self.up2(h_up3), h2, c2), fused_emb)
        h_up1 = self.up_block1(match_and_cat(self.up1(h_up2), h1, c1), fused_emb)
        h_up0 = self.up_block0(match_and_cat(self.up0(h_up1), h0, c0), fused_emb)

        return self.outc(h_up0)

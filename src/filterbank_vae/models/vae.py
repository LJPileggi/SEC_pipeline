import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_BOTTLENECK_TYPE = "2d"


class AttentionBlock(nn.Module):
    """
    Multi-Head Self-Attention leggera per correlare armoniche distanti
    e dipendenze temporali globali a bassa risoluzione.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_groups=min(32, channels // 4), num_channels=channels)
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        tokens = h.permute(0, 2, 3, 1).reshape(B, H * W, C)
        attn_out, _ = self.mha(tokens, tokens, tokens)
        attn_out = attn_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + attn_out


class ResBlock(nn.Module):
    """
    Blocco residuo convoluzionale con GroupNorm e SiLU.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(32, in_channels // 4), num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_channels // 4), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.shortcut(x) + h


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=256, commitment_cost=0.25, decay=0.99, epsilon=1e-5, threshold_dead=1.0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.threshold_dead = threshold_dead

        # Inizializzazione normalizzata
        embedding = torch.randn(embedding_dim, num_embeddings)
        embedding = F.normalize(embedding, p=2, dim=0)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_cluster_size", torch.ones(num_embeddings))
        self.register_buffer("ema_w", embedding.clone())

    def forward(self, inputs):
        # inputs: [B, C, H, W] -> [B, H, W, C]
        inputs_permuted = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs_permuted.view(-1, self.embedding_dim)

        # 1. Normalizzazione L2 dei vettori per evitare divergenze di scala
        flat_input_norm = F.normalize(flat_input, p=2, dim=-1)
        embedding_norm = F.normalize(self.embedding, p=2, dim=0)

        # Distanza euclidea su ipersfera
        distances = (
            torch.sum(flat_input_norm ** 2, dim=1, keepdim=True)
            + torch.sum(embedding_norm ** 2, dim=0, keepdim=True)
            - 2 * torch.matmul(flat_input_norm, embedding_norm)
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.t()).view(inputs_permuted.shape)

        if self.training:
            target_dtype = self.ema_w.dtype
            flat_input_fp32 = flat_input.to(dtype=target_dtype)
            encodings_fp32 = encodings.to(dtype=target_dtype)

            cluster_size = torch.sum(encodings_fp32, dim=0)
            self.ema_cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

            dw = torch.matmul(flat_input_fp32.t(), encodings_fp32)
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            # Riassegnazione dei codici morti (restart garantito in Float32)
            dead_codes = self.ema_cluster_size < self.threshold_dead
            if dead_codes.any():
                n_dead = dead_codes.sum().item()
                rand_indices = torch.randint(0, flat_input_fp32.size(0), (n_dead,), device=inputs.device)
                sampled_latents = flat_input_fp32[rand_indices].t()
                self.ema_w.data[:, dead_codes] = sampled_latents
                self.ema_cluster_size.data[dead_codes] = self.threshold_dead

            n = torch.sum(self.ema_cluster_size)
            smoothed_cluster_size = (
                (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            )
            normalized_w = self.ema_w / smoothed_cluster_size.unsqueeze(0)
            self.embedding.data.copy_(normalized_w)

        e_latent_loss = F.mse_loss(inputs_permuted, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        quantized = inputs_permuted + (quantized - inputs_permuted).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return quantized, loss, encoding_indices.view(inputs.shape[0], inputs.shape[2], inputs.shape[3])

    def quantize_code(self, encoding_indices):
        B, H, W = encoding_indices.shape
        flat_indices = encoding_indices.view(-1, 1)
        encodings = torch.zeros(flat_indices.shape[0], self.num_embeddings, device=encoding_indices.device)
        encodings.scatter_(1, flat_indices, 1)
        quantized = torch.matmul(encodings, self.embedding.t()).view(B, H, W, self.embedding_dim)
        return quantized.permute(0, 3, 1, 2).contiguous()


# ==============================================================================
# 1. RAMO MEL VQ-VAE (Input: [B, 1, 64, 1152])
# ==============================================================================

class MelEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, latent_dim=256):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.res1 = ResBlock(base_channels, base_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)

        self.res2 = ResBlock(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)

        self.res3 = ResBlock(base_channels * 4, base_channels * 4)
        self.down3 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1)

        self.res4 = ResBlock(base_channels * 4, base_channels * 4)
        self.down4 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1)

        self.attn = AttentionBlock(base_channels * 4, num_heads=4)
        self.proj = nn.Conv2d(base_channels * 4, latent_dim, kernel_size=1)

    def forward(self, x):
        h = self.init_conv(x)
        h = self.down1(self.res1(h))
        h = self.down2(self.res2(h))
        h = self.down3(self.res3(h))
        h = self.down4(self.res4(h))
        h = self.attn(h)
        return self.proj(h)


class MelDecoder(nn.Module):
    def __init__(self, out_channels=1, base_channels=64, latent_dim=256):
        super().__init__()
        self.proj_in = nn.Conv2d(latent_dim, base_channels * 4, kernel_size=1)
        self.attn = AttentionBlock(base_channels * 4, num_heads=4)

        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.res1 = ResBlock(base_channels * 4, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.res2 = ResBlock(base_channels * 2, base_channels * 2)

        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.res3 = ResBlock(base_channels, base_channels)

        self.up4 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.res4 = ResBlock(base_channels, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        h = self.proj_in(z)
        h = self.attn(h)
        h = self.res1(self.up1(h))
        h = self.res2(self.up2(h))
        h = self.res3(self.up3(h))
        h = self.res4(self.up4(h))
        return self.out_conv(h)


class MelVQVAE(nn.Module):
    def __init__(self, base_channels=64, latent_dim=256, num_embeddings=1024):
        super().__init__()
        self.encoder = MelEncoder(in_channels=1, base_channels=base_channels, latent_dim=latent_dim)
        self.quantizer = VectorQuantizerEMA(num_embeddings=num_embeddings, embedding_dim=latent_dim)
        self.decoder = MelDecoder(out_channels=1, base_channels=base_channels, latent_dim=latent_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z_e)
        recon = self.decoder(z_q)
        return recon, vq_loss, z_e, z_q, indices


# ==============================================================================
# 2. RAMO OCTAVE VQ-VAE (Input: [B, 1, 320, 1152] via PCHIP GPU)
# ==============================================================================

class OctaveEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, latent_dim=256):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.res1 = ResBlock(base_channels, base_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)

        self.res2 = ResBlock(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)

        self.res3 = ResBlock(base_channels * 4, base_channels * 4)
        self.down3 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1)

        self.res4 = ResBlock(base_channels * 4, base_channels * 4)
        self.down4 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1)

        self.res5 = ResBlock(base_channels * 4, base_channels * 4)
        self.down5 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=(2, 1), padding=1)

        self.align_pool = nn.AdaptiveAvgPool2d((4, 72))
        self.res_final = ResBlock(base_channels * 4, base_channels * 4)

        self.attn = AttentionBlock(base_channels * 4, num_heads=4)
        self.proj = nn.Conv2d(base_channels * 4, latent_dim, kernel_size=1)

    def forward(self, x):
        h = self.init_conv(x)
        h = self.down1(self.res1(h))
        h = self.down2(self.res2(h))
        h = self.down3(self.res3(h))
        h = self.down4(self.res4(h))
        h = self.down5(self.res5(h))
        h = self.res_final(self.align_pool(h))
        h = self.attn(h)
        return self.proj(h)


class OctaveDecoder(nn.Module):
    def __init__(self, out_channels=1, base_channels=64, latent_dim=256, target_freq=320, target_time=1152):
        super().__init__()
        self.target_freq = target_freq
        self.target_time = target_time

        self.proj_in = nn.Conv2d(latent_dim, base_channels * 4, kernel_size=1)
        self.attn = AttentionBlock(base_channels * 4, num_heads=4)

        self.up5 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1))
        self.res5 = ResBlock(base_channels * 4, base_channels * 4)

        self.up4 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.res4 = ResBlock(base_channels * 4, base_channels * 4)

        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.res3 = ResBlock(base_channels * 4, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.res2 = ResBlock(base_channels * 2, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.res1 = ResBlock(base_channels, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        h = self.proj_in(z)
        h = self.attn(h)
        h = self.res5(self.up5(h))
        h = self.res4(self.up4(h))
        h = self.res3(self.up3(h))
        h = self.res2(self.up2(h))
        h = self.res1(self.up1(h))
        out = self.out_conv(h)

        if out.shape[-2:] != (self.target_freq, self.target_time):
            out = F.interpolate(out, size=(self.target_freq, self.target_time), mode='bilinear', align_corners=False)
        return out


class OctaveVQVAE(nn.Module):
    def __init__(self, base_channels=64, latent_dim=256, num_embeddings=1024, shared_quantizer=None):
        super().__init__()
        self.encoder = OctaveEncoder(in_channels=1, base_channels=base_channels, latent_dim=latent_dim)
        if shared_quantizer is not None:
            self.quantizer = shared_quantizer
        else:
            self.quantizer = VectorQuantizerEMA(num_embeddings=num_embeddings, embedding_dim=latent_dim)
        self.decoder = OctaveDecoder(out_channels=1, base_channels=base_channels, latent_dim=latent_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z_e)
        recon = self.decoder(z_q)
        return recon, vq_loss, z_e, z_q, indices


# ==============================================================================
# 3. DISCRIMINATORE PATCHGAN CON FEATURE EXTRACTION (PyTorch Puro, Anti-Inplace)
# ==============================================================================

class DiscriminatorWithFeatures(nn.Module):
    """
    Discriminatore convoluzionale a Patch privo di BatchNorm e di operazioni in-place.
    Usa GroupNorm per evitare il tracciamento di running_mean che mandava in crash autograd.
    """
    def __init__(self, in_channels=1, ndf=64):
        super().__init__()
        # inplace=False tassativo per non sporcare i tensori salvati per il backward
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, (ndf * 2) // 4), num_channels=ndf * 2),
            nn.LeakyReLU(0.2, inplace=False)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, (ndf * 4) // 4), num_channels=ndf * 4),
            nn.LeakyReLU(0.2, inplace=False)
        )
        self.out_conv = nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        features = []
        h = self.block1(x)
        features.append(h)
        h = self.block2(h)
        features.append(h)
        h = self.block3(h)
        features.append(h)
        out = self.out_conv(h)
        return out, features


MelVAE = MelVQVAE
OctaveVAE = OctaveVQVAE

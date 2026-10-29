import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_BOTTLENECK_TYPE = "rvq_2d"


class TemporalClassAuxiliaryHead(nn.Module):
    """
    Testa ausiliaria di supervisione semantica spazio-temporale su griglia [8, 144].
    Esegue pooling sulle 8 bande frequenziali preservando i 144 step temporali.
    """
    def __init__(self, latent_dim=256, num_classes=22, dropout=0.1):
        super().__init__()
        in_dim = latent_dim * 2  # avg + max su frequenza
        hidden_dim = latent_dim

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(32, hidden_dim // 4), num_channels=hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(32, hidden_dim // 4), num_channels=hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout)
        )
        self.classifier = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, z_q):
        avg_f = torch.mean(z_q, dim=2)  # [B, D, 144]
        max_f = torch.amax(z_q, dim=2)  # [B, D, 144]
        feat = torch.cat([avg_f, max_f], dim=1)  # [B, 2D, 144]

        h = self.temporal_conv(feat)
        return self.classifier(h)  # [B, num_classes, 144]


ClassAuxiliaryHead = TemporalClassAuxiliaryHead


class AttentionBlock(nn.Module):
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
    def __init__(self, num_embeddings=1024, embedding_dim=256, commitment_cost=0.05, decay=0.99, epsilon=1e-5, threshold_dead=1.0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.threshold_dead = threshold_dead

        embedding = torch.randn(embedding_dim, num_embeddings)
        embedding = F.normalize(embedding, p=2, dim=0)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_cluster_size", torch.ones(num_embeddings))
        self.register_buffer("ema_w", embedding.clone())

    def set_decay(self, new_decay):
        self.decay = new_decay

    def get_codebook_similarities(self, inputs):
        inputs_permuted = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs_permuted.view(-1, self.embedding_dim)
        flat_input_norm = F.normalize(flat_input, p=2, dim=-1)
        embedding_norm = F.normalize(self.embedding.to(dtype=flat_input.dtype), p=2, dim=0)
        similarities = torch.matmul(flat_input_norm, embedding_norm)
        return similarities, inputs_permuted.shape

    def forward(self, inputs, update_ema=True):
        inputs_permuted = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs_permuted.view(-1, self.embedding_dim)

        flat_input_norm = F.normalize(flat_input, p=2, dim=-1)
        embedding_norm = F.normalize(self.embedding.to(dtype=flat_input.dtype), p=2, dim=0)

        distances = (
            torch.sum(flat_input_norm ** 2, dim=1, keepdim=True)
            + torch.sum(embedding_norm ** 2, dim=0, keepdim=True)
            - 2 * torch.matmul(flat_input_norm, embedding_norm)
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device, dtype=flat_input.dtype)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.to(dtype=flat_input.dtype).t()).view(inputs_permuted.shape)

        if self.training and update_ema:
            target_dtype = self.ema_w.dtype
            flat_input_fp32 = flat_input.to(dtype=target_dtype)
            encodings_fp32 = encodings.to(dtype=target_dtype)

            cluster_size = torch.sum(encodings_fp32, dim=0)
            self.ema_cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

            dw = torch.matmul(flat_input_fp32.t(), encodings_fp32)
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

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


class ResidualVectorQuantizerEMA(nn.Module):
    def __init__(self, num_quantizers=4, num_embeddings=1024, embedding_dim=256, commitment_cost=0.05, decay=0.99):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.layers = nn.ModuleList([
            VectorQuantizerEMA(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost,
                decay=decay
            )
            for _ in range(num_quantizers)
        ])

    def set_decay(self, new_decay):
        for layer in self.layers:
            layer.set_decay(new_decay)

    def forward(self, inputs, update_ema=True):
        residual = inputs
        quantized_out = torch.zeros_like(inputs)
        total_loss = torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)
        all_indices = []

        for layer in self.layers:
            q_stage, loss_stage, indices_stage = layer(residual, update_ema=update_ema)
            residual = residual - q_stage
            quantized_out = quantized_out + q_stage
            total_loss = total_loss + loss_stage
            all_indices.append(indices_stage)

        indices = torch.stack(all_indices, dim=1)
        return quantized_out, total_loss, indices

    def quantize_code(self, encoding_indices):
        quantized_out = 0.0
        for i, layer in enumerate(self.layers):
            stage_indices = encoding_indices[:, i, :, :]
            quantized_out = quantized_out + layer.quantize_code(stage_indices)
        return quantized_out


class MelEncoder(nn.Module):
    """
    Downsampling da [64, 1152] a [8, 144] (Downsampling 8x simmetrico):
    - down1: 64x1152 -> 32x576 (stride=2)
    - down2: 32x576  -> 16x288 (stride=2)
    - down3: 16x288  -> 8x144  (stride=2)
    - down4: 8x144   -> 8x144  (stride=1, preserva la griglia fine)
    """
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
        self.down4 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1)

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
    """
    Upsampling da [8, 144] a [64, 1152] (Upsampling 8x simmetrico):
    - up1: 8x144   -> 8x144   (stride=1)
    - up2: 8x144   -> 16x288  (stride=2)
    - up3: 16x288  -> 32x576  (stride=2)
    - up4: 32x576  -> 64x1152 (stride=2)
    """
    def __init__(self, out_channels=1, base_channels=64, latent_dim=256):
        super().__init__()
        self.proj_in = nn.Conv2d(latent_dim, base_channels * 4, kernel_size=1)
        self.attn = AttentionBlock(base_channels * 4, num_heads=4)

        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1)
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
    def __init__(self, base_channels=64, latent_dim=256, num_embeddings=1024, num_quantizers=4, commitment_cost=0.05):
        super().__init__()
        self.encoder = MelEncoder(in_channels=1, base_channels=base_channels, latent_dim=latent_dim)
        self.quantizer = ResidualVectorQuantizerEMA(
            num_quantizers=num_quantizers,
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost
        )
        self.decoder = MelDecoder(out_channels=1, base_channels=base_channels, latent_dim=latent_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z_e, update_ema=True)
        recon = self.decoder(z_q)
        return recon, vq_loss, z_e, z_q, indices


class OctaveEncoder(nn.Module):
    """
    Downsampling da [320, 1152] a [8, 144]:
    - down1: 320x1152 -> 160x576 (stride=2)
    - down2: 160x576  -> 80x288  (stride=2)
    - down3: 80x288   -> 40x144  (stride=2)
    - down4: 40x144   -> 20x144  (stride=(2, 1), preserva il tempo a 144)
    - down5: 20x144   -> 10x144  (stride=(2, 1), preserva il tempo a 144)
    - align_pool: (8, 144) esatto
    """
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
        self.down4 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=(2, 1), padding=1)

        self.res5 = ResBlock(base_channels * 4, base_channels * 4)
        self.down5 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=(2, 1), padding=1)

        self.align_pool = nn.AdaptiveAvgPool2d((8, 144))
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
    """
    Upsampling simmetrico da [8, 144] a [320, 1152]:
    - up5: 8x144   -> 16x144  (stride=(2, 1))
    - up4: 16x144  -> 32x144  (stride=(2, 1))
    - up3: 32x144  -> 64x288  (stride=2)
    - up2: 64x288  -> 128x576 (stride=2)
    - up1: 128x576 -> 256x1152 (stride=2)
    Interpolazione finale garantita su (320, 1152).
    """
    def __init__(self, out_channels=1, base_channels=64, latent_dim=256, target_freq=320, target_time=1152):
        super().__init__()
        self.target_freq = target_freq
        self.target_time = target_time

        self.proj_in = nn.Conv2d(latent_dim, base_channels * 4, kernel_size=1)
        self.attn = AttentionBlock(base_channels * 4, num_heads=4)

        self.up5 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1))
        self.res5 = ResBlock(base_channels * 4, base_channels * 4)

        self.up4 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1))
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
    def __init__(self, base_channels=64, latent_dim=256, num_embeddings=1024, num_quantizers=4, commitment_cost=0.05, shared_quantizer=None):
        super().__init__()
        self.encoder = OctaveEncoder(in_channels=1, base_channels=base_channels, latent_dim=latent_dim)
        if shared_quantizer is not None:
            self.quantizer = shared_quantizer
        else:
            self.quantizer = ResidualVectorQuantizerEMA(
                num_quantizers=num_quantizers,
                num_embeddings=num_embeddings,
                embedding_dim=latent_dim,
                commitment_cost=commitment_cost
            )
        self.decoder = OctaveDecoder(out_channels=1, base_channels=base_channels, latent_dim=latent_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z_e, update_ema=False)
        recon = self.decoder(z_q)
        return recon, vq_loss, z_e, z_q, indices


class DiscriminatorWithFeatures(nn.Module):
    def __init__(self, in_channels=1, ndf=64, num_classes=22):
        super().__init__()
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

        self.class_emb = nn.Embedding(num_classes, ndf * 4)
        nn.init.normal_(self.class_emb.weight, std=0.02)

    def forward(self, x, class_idx=None):
        features = []
        h = self.block1(x)
        features.append(h)
        h = self.block2(h)
        features.append(h)
        h = self.block3(h)
        features.append(h)
        out = self.out_conv(h)

        if class_idx is not None:
            h_pool = torch.mean(h, dim=(2, 3))
            c_vec = self.class_emb(class_idx)
            scale = 1.0 / (c_vec.shape[-1] ** 0.5)
            proj_term = (torch.sum(h_pool * c_vec, dim=1, keepdim=True) * scale)[:, :, None, None]
            out = out + proj_term

        return out, features


MelVAE = MelVQVAE
OctaveVAE = OctaveVQVAE

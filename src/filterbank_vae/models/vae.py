import torch
import torch.nn as nn
import torch.nn.functional as F

# Configurazione globale di default per la tipologia di bottleneck: "1d" oppure "2d"
DEFAULT_BOTTLENECK_TYPE = "1d"


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
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        h = self.norm(x)
        # Reshape in sequenza di token: [B, H*W, C]
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


# ==============================================================================
# 1. RAMO MEL VAE (Input: [B, 1, 64, 1152])
# ==============================================================================

class MelEncoder(nn.Module):
    """
    Encoder per Mel Spectrograms:
    4 stadi di dimezzamento:
    64x1152 -> 32x576 -> 16x288 -> 8x144 -> 4x72
    """
    def __init__(self, in_channels=1, base_channels=64, bottleneck_type=DEFAULT_BOTTLENECK_TYPE, latent_dim=512):
        super().__init__()
        self.bottleneck_type = bottleneck_type
        self.latent_dim = latent_dim

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Stadio 1: 64x1152 -> 32x576 (C: 64)
        self.res1 = ResBlock(base_channels, base_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)

        # Stadio 2: 32x576 -> 16x288 (C: 128)
        self.res2 = ResBlock(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)

        # Stadio 3: 16x288 -> 8x144 (C: 256)
        self.res3 = ResBlock(base_channels * 4, base_channels * 4)
        self.down3 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1)

        # Stadio 4: 8x144 -> 4x72 (C: 256)
        self.res4 = ResBlock(base_channels * 4, base_channels * 4)
        self.down4 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1)

        # Attenzione globale al termine del downsampling
        self.attn = AttentionBlock(base_channels * 4, num_heads=4)

        if self.bottleneck_type == "1d":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc_mu = nn.Linear(base_channels * 4, latent_dim)
            self.fc_logvar = nn.Linear(base_channels * 4, latent_dim)
        else: # "2d"
            self.conv_mu = nn.Conv2d(base_channels * 4, latent_dim, kernel_size=1)
            self.conv_logvar = nn.Conv2d(base_channels * 4, latent_dim, kernel_size=1)

    def forward(self, x):
        h = self.init_conv(x)
        h = self.down1(self.res1(h))
        h = self.down2(self.res2(h))
        h = self.down3(self.res3(h))
        h = self.down4(self.res4(h))
        h = self.attn(h)

        if self.bottleneck_type == "1d":
            flat = self.pool(h).flatten(1)
            mu = self.fc_mu(flat)
            logvar = self.fc_logvar(flat)
        else:
            mu = self.conv_mu(h)
            logvar = self.conv_logvar(h)
        return mu, logvar


class MelDecoder(nn.Module):
    """
    Decoder per Mel Spectrograms:
    4 stadi di raddoppio speculari:
    4x72 -> 8x144 -> 16x288 -> 32x576 -> 64x1152
    """
    def __init__(self, out_channels=1, base_channels=64, bottleneck_type=DEFAULT_BOTTLENECK_TYPE, latent_dim=512):
        super().__init__()
        self.bottleneck_type = bottleneck_type
        self.latent_dim = latent_dim

        if self.bottleneck_type == "1d":
            self.fc_in = nn.Linear(latent_dim, base_channels * 4 * 4 * 72)
        else:
            self.conv_in = nn.Conv2d(latent_dim, base_channels * 4, kernel_size=1)

        self.attn = AttentionBlock(base_channels * 4, num_heads=4)

        # 4x72 -> 8x144
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.res1 = ResBlock(base_channels * 4, base_channels * 4)

        # 8x144 -> 16x288
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.res2 = ResBlock(base_channels * 2, base_channels * 2)

        # 16x288 -> 32x576
        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.res3 = ResBlock(base_channels, base_channels)

        # 32x576 -> 64x1152
        self.up4 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.res4 = ResBlock(base_channels, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        if self.bottleneck_type == "1d":
            h = self.fc_in(z).view(-1, 256, 4, 72)
        else:
            h = self.conv_in(z)

        h = self.attn(h)
        h = self.res1(self.up1(h))
        h = self.res2(self.up2(h))
        h = self.res3(self.up3(h))
        h = self.res4(self.up4(h))
        out = self.out_conv(h)
        return out


class MelVAE(nn.Module):
    def __init__(self, base_channels=64, bottleneck_type=DEFAULT_BOTTLENECK_TYPE, latent_dim=512):
        super().__init__()
        self.encoder = MelEncoder(in_channels=1, base_channels=base_channels, bottleneck_type=bottleneck_type, latent_dim=latent_dim)
        self.decoder = MelDecoder(out_channels=1, base_channels=base_channels, bottleneck_type=bottleneck_type, latent_dim=latent_dim)
        self.bottleneck_type = bottleneck_type

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z


# ==============================================================================
# 2. RAMO OCTAVE VAE (Input: [B, 1, 320, 1152] via PCHIP GPU)
# ==============================================================================

class OctaveEncoder(nn.Module):
    """
    Encoder per Spettrogrammi a frazioni d'ottava (32esime):
    Ingresso a 320 bande.
    Comprime le frequenze da 320 a 4 (rapporto 80:1) e il tempo da 1152 a 72 (rapporto 16:1):
    320x1152 -> (s: 2x2) 160x576 -> (s: 2x2) 80x288 -> (s: 2x2) 40x144 -> 
                 (s: 2x2) 20x72  -> (s: 2x1) 10x72  -> (s: 2x1) 5x72 -> (s: Conv 5->4) 4x72
    """
    def __init__(self, in_channels=1, base_channels=64, bottleneck_type=DEFAULT_BOTTLENECK_TYPE, latent_dim=512):
        super().__init__()
        self.bottleneck_type = bottleneck_type
        self.latent_dim = latent_dim

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # 320x1152 -> 160x576
        self.res1 = ResBlock(base_channels, base_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)

        # 160x576 -> 80x288
        self.res2 = ResBlock(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)

        # 80x288 -> 40x144
        self.res3 = ResBlock(base_channels * 4, base_channels * 4)
        self.down3 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1)

        # 40x144 -> 20x72 (Tempo è ora a 72, lo preserviamo!)
        self.res4 = ResBlock(base_channels * 4, base_channels * 4)
        self.down4 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1)

        # 20x72 -> 10x72 (Stride 2 solo in frequenza)
        self.res5 = ResBlock(base_channels * 4, base_channels * 4)
        self.down5 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=(2, 1), padding=1)

        # 10x72 -> 4x72 (Conv che mappa esattamente 10 bin a 4 bin)
        self.res6 = ResBlock(base_channels * 4, base_channels * 4)
        self.down6 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=(4, 3), stride=(2, 1), padding=(0, 1))

        self.attn = AttentionBlock(base_channels * 4, num_heads=4)

        if self.bottleneck_type == "1d":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc_mu = nn.Linear(base_channels * 4, latent_dim)
            self.fc_logvar = nn.Linear(base_channels * 4, latent_dim)
        else:
            self.conv_mu = nn.Conv2d(base_channels * 4, latent_dim, kernel_size=1)
            self.conv_logvar = nn.Conv2d(base_channels * 4, latent_dim, kernel_size=1)

    def forward(self, x):
        h = self.init_conv(x)
        h = self.down1(self.res1(h))
        h = self.down2(self.res2(h))
        h = self.down3(self.res3(h))
        h = self.down4(self.res4(h))
        h = self.down5(self.res5(h))
        h = self.down6(self.res6(h))
        h = self.attn(h)

        if self.bottleneck_type == "1d":
            flat = self.pool(h).flatten(1)
            mu = self.fc_mu(flat)
            logvar = self.fc_logvar(flat)
        else:
            mu = self.conv_mu(h)
            logvar = self.conv_logvar(h)
        return mu, logvar


class OctaveDecoder(nn.Module):
    """
    Decoder per Spettrogrammi d'ottava (ricostruisce [B, 1, 320, 1152]):
    4x72 -> 10x72 -> 20x72 -> 40x144 -> 80x288 -> 160x576 -> 320x1152
    """
    def __init__(self, out_channels=1, base_channels=64, bottleneck_type=DEFAULT_BOTTLENECK_TYPE, latent_dim=512):
        super().__init__()
        self.bottleneck_type = bottleneck_type
        self.latent_dim = latent_dim

        if self.bottleneck_type == "1d":
            self.fc_in = nn.Linear(latent_dim, base_channels * 4 * 4 * 72)
        else:
            self.conv_in = nn.Conv2d(latent_dim, base_channels * 4, kernel_size=1)

        self.attn = AttentionBlock(base_channels * 4, num_heads=4)

        # 4x72 -> 10x72
        self.up6 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=(4, 3), stride=(2, 1), padding=(0, 1))
        self.res6 = ResBlock(base_channels * 4, base_channels * 4)

        # 10x72 -> 20x72
        self.up5 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1))
        self.res5 = ResBlock(base_channels * 4, base_channels * 4)

        # 20x72 -> 40x144
        self.up4 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.res4 = ResBlock(base_channels * 4, base_channels * 4)

        # 40x144 -> 80x288
        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.res3 = ResBlock(base_channels * 4, base_channels * 4)

        # 80x288 -> 160x576
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.res2 = ResBlock(base_channels * 2, base_channels * 2)

        # 160x576 -> 320x1152
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.res1 = ResBlock(base_channels, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        if self.bottleneck_type == "1d":
            h = self.fc_in(z).view(-1, 256, 4, 72)
        else:
            h = self.conv_in(z)

        h = self.attn(h)
        h = self.res6(self.up6(h))
        h = self.res5(self.up5(h))
        h = self.res4(self.up4(h))
        h = self.res3(self.up3(h))
        h = self.res2(self.up2(h))
        h = self.res1(self.up1(h))
        out = self.out_conv(h)
        return out


class OctaveVAE(nn.Module):
    def __init__(self, base_channels=64, bottleneck_type=DEFAULT_BOTTLENECK_TYPE, latent_dim=512):
        super().__init__()
        self.encoder = OctaveEncoder(in_channels=1, base_channels=base_channels, bottleneck_type=bottleneck_type, latent_dim=latent_dim)
        self.decoder = OctaveDecoder(out_channels=1, base_channels=base_channels, bottleneck_type=bottleneck_type, latent_dim=latent_dim)
        self.bottleneck_type = bottleneck_type

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z

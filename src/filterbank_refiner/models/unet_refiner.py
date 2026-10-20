import torch
import torch.nn as nn
import torch.nn.functional as F


class RefinerResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(16, in_channels // 2), num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(16, out_channels // 2), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.shortcut(x) + h


class MelRefinerUNet(nn.Module):
    """
    Direct Residual Generator: prende il Mel grezzo cross-domain x_cond [B, 1, 64, 1152]
    e genera direttamente la rifinitura ad alta frequenza in 1 singolo step.
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=32, channel_mult=(1, 2, 4)):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        c1 = base_channels * channel_mult[0]  # 32
        c2 = base_channels * channel_mult[1]  # 64
        c3 = base_channels * channel_mult[2]  # 128

        # Encoder
        self.enc1 = RefinerResBlock(base_channels, c1)
        self.down1 = nn.Conv2d(c1, c1, kernel_size=4, stride=2, padding=1)  # -> [32, 576]

        self.enc2 = RefinerResBlock(c1, c2)
        self.down2 = nn.Conv2d(c2, c2, kernel_size=4, stride=2, padding=1)  # -> [16, 288]

        self.enc3 = RefinerResBlock(c2, c3)
        self.down3 = nn.Conv2d(c3, c3, kernel_size=4, stride=2, padding=1)  # -> [8, 144]

        # Bottleneck
        self.mid1 = RefinerResBlock(c3, c3)
        self.mid2 = RefinerResBlock(c3, c3)

        # Decoder con skip connections
        self.up3 = nn.ConvTranspose2d(c3, c3, kernel_size=4, stride=2, padding=1)
        self.dec3 = RefinerResBlock(c3 + c3, c2)

        self.up2 = nn.ConvTranspose2d(c2, c2, kernel_size=4, stride=2, padding=1)
        self.dec2 = RefinerResBlock(c2 + c2, c1)

        self.up1 = nn.ConvTranspose2d(c1, c1, kernel_size=4, stride=2, padding=1)
        self.dec1 = RefinerResBlock(c1 + c1, base_channels)

        self.out_norm = nn.GroupNorm(num_groups=min(16, base_channels // 2), num_channels=base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x_cond):
        h0 = self.init_conv(x_cond)

        h1 = self.enc1(h0)
        d1 = self.down1(h1)

        h2 = self.enc2(d1)
        d2 = self.down2(h2)

        h3 = self.enc3(d2)
        d3 = self.down3(h3)

        m = self.mid1(d3)
        m = self.mid2(m)

        u3 = self.up3(m)
        u3 = self.dec3(torch.cat([u3, h3], dim=1))

        u2 = self.up2(u3)
        u2 = self.dec2(torch.cat([u2, h2], dim=1))

        u1 = self.up1(u2)
        u1 = self.dec1(torch.cat([u1, h1], dim=1))

        residual = self.out_conv(F.silu(self.out_norm(u1)))
        
        # Connessione residua globale: la rete predice il differenziale di iperrisoluzione
        return x_cond + residual

    @torch.no_grad()
    def sample_refined(self, x_cond, num_steps=None):
        """Inferenza deterministica a 1-step: zero loop numerici, pura forward pass."""
        return self.forward(x_cond)


class RefinerPatchDiscriminator(nn.Module):
    """
    Discriminatore multi-scala/patch per spettrogrammi [B, 1, 64, 1152].
    Estrae feature intermedie per la feature matching loss.
    """
    def __init__(self, in_channels=1, ndf=32):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(16, (ndf * 2) // 2), num_channels=ndf * 2),
            nn.LeakyReLU(0.2, inplace=False)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(16, (ndf * 4) // 2), num_channels=ndf * 4),
            nn.LeakyReLU(0.2, inplace=False)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(16, (ndf * 4) // 2), num_channels=ndf * 4),
            nn.LeakyReLU(0.2, inplace=False)
        )
        self.out_conv = nn.Conv2d(ndf * 4, 1, kernel_size=3, padding=1)

    def forward(self, x):
        features = []
        h = self.layer1(x)
        features.append(h)
        h = self.layer2(h)
        features.append(h)
        h = self.layer3(h)
        features.append(h)
        h = self.layer4(h)
        features.append(h)
        out = self.out_conv(h)
        return out, features

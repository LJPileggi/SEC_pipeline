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
    Feature-wise Linear Modulation block to inject semantic labels and temporal embeddings
    deeply into the U-Net feature maps.
    """
    def __init__(self, embedding_dim, channels):
        super().__init__()
        # Predicts gamma (scaling) and beta (shifting) coefficients for each channel
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, channels * 2)
        )

    def forward(self, x, emb):
        # emb shape: [B, embedding_dim], x shape: [B, C, F, T]
        emb_out = self.mlp(emb).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = emb_out.chunk(2, dim=1)
        return gamma * x + beta # Modulates normalized or raw activations lineary

class AsymmetricConvBlock(nn.Module):
    """
    U-Net convolutional block substituting standard square kernels with asymmetric 
    rectangular kernels (1x7 for time, 7x1 for frequency) to isolate structural deformations.
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

class ConditionalUNet(nn.Module):
    """
    Conditional U-Net architecture optimized for high-resolution Log-Mel restoration.
    Accepts noisy spectrograms concatenated with low-resolution filterbanks as a rigid spatial anchor.
    """
    def __init__(self, num_classes=22, base_channels=64, emb_dim=256):
        super().__init__()
        self.num_classes = num_classes
        
        # 23 slots: indices 0-21 represent core acoustic labels, index 22 holds the "Unknown" null token for CFG
        self.class_embedding = nn.Embedding(num_classes + 1, emb_dim)
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU()
        )
        
        # Combined condition mapping used to feed the sequential FiLM injectors
        self.comb_emb_projector = nn.Linear(emb_dim * 2, emb_dim)

        # 🎯 INPUT CHANNELS = 2: Channel 0 is x_t (Noisy Mel), Channel 1 is C (Spectral anchor)
        self.inc = AsymmetricConvBlock(2, base_channels, emb_dim)
        
        # Downsampling Encoder blocks
        self.down1 = nn.Sequential(nn.MaxPool2d((2, 2)), AsymmetricConvBlock(base_channels, base_channels * 2, emb_dim))
        self.down2 = nn.Sequential(nn.MaxPool2d((2, 2)), AsymmetricConvBlock(base_channels * 2, base_channels * 4, emb_dim))
        
        # Bottleneck
        self.mid = AsymmetricConvBlock(base_channels * 4, base_channels * 4, emb_dim)
        
        # Upsampling Decoder blocks
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.up_block1 = AsymmetricConvBlock(base_channels * 4, base_channels * 2, emb_dim) # Cat channels: (base*2 + base*2)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.up_block2 = AsymmetricConvBlock(base_channels * 2, base_channels, emb_dim) # Cat channels: (base + base)
        
        # Final projection layer returning pure predicted noise epsilon matching input shape (1 channel)
        self.outc = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x_t, t, conditioning_C, class_labels):
        """
        args:
         - x_t: Noisy Mel tensor [B, 1, 64, 700]
         - t: Diffusion timesteps integers [B]
         - conditioning_C: Spectral mask envelope tensor [B, 1, 64, 700]
         - class_labels: Numerical semantic indices [B]
        """
        # 1. Spatial conditioning via direct channel-wise concatenation
        # Shapes check: [B, 1, 64, 700] concat [B, 1, 64, 700] -> [B, 2, 64, 700]
        x_in = torch.cat([x_t, conditioning_C], dim=1)

        # 2. Extract and concatenate semantic and temporal dense conditions
        t_emb = self.time_embedding(t)                 # [B, emb_dim]
        c_emb = self.class_embedding(class_labels)       # [B, emb_dim]
        
        # Core fused embedding propagated to all FiLM layers
        fused_emb = self.comb_emb_projector(torch.cat([t_emb, c_emb], dim=-1)) # [B, emb_dim]

        # 3. Downward structural path
        h0 = self.inc(x_in, fused_emb)                 # [B, base, 64, 700]
        h1 = self.down1[1](self.down1[0](h0), fused_emb) # [B, base*2, 32, 350]
        h2 = self.down2[1](self.down2[0](h1), fused_emb) # [B, base*4, 16, 175]

        # 4. Bottleneck representation
        h_mid = self.mid(h2, fused_emb)

        # 5. Upward restorative path incorporating skip connections
        u1 = self.up1(h_mid)
        # Interpolation guard to prevent rounding mismatches on odd sizes (e.g. 175 * 2 = 350)
        if u1.shape[-1] != h1.shape[-1] or u1.shape[-2] != h1.shape[-2]:
            u1 = F.interpolate(u1, size=(h1.shape[-2], h1.shape[-1]), mode='bilinear', align_corners=False)
        h_up1 = self.up_block1(torch.cat([u1, h1], dim=1), fused_emb)

        u2 = self.up2(h_up1)
        if u2.shape[-1] != h0.shape[-1] or u2.shape[-2] != h0.shape[-2]:
            u2 = F.interpolate(u2, size=(h0.shape[-2], h0.shape[-1]), mode='bilinear', align_corners=False)
        h_up2 = self.up_block2(torch.cat([u2, h0], dim=1), fused_emb)

        return self.outc(h_up2) # Out shape matches noise target exactly: [B, 1, 64, 700]

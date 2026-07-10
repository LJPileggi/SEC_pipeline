import os
import sys
import torch
import nn = torch.nn
import F = torch.nn.functional

# Dynamic root injection to safely import core production modules from src/
current_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

# Import production processing components directly from core modules
from models import spectrogram_n_octaveband_generator_gpu, convert_octave_to_msclap_mel

class OnlineSpectrogramPipeline(nn.Module):
    """
    GPU-accelerated spectral transformation engine. Isolates and loads the HTS-AT backbone 
    directly from CLAP state_dict checkpoints, bypassing RoBERTa initialization.
    """
    def __init__(self, weights_path, sample_rate=51200, device='cuda'):
        super().__init__()
        self.sample_rate = sample_rate
        
        # Explicit import of the underlying standalone HTS-AT Swin-Transformer structure from msclap
        from msclap.models.htsat import HTSAT_Swin_Transformer
        
        # Instantiate the pure standalone backbone with Microsoft factory parameters
        self.htsat = HTSAT_Swin_Transformer(
            spec_size=256,
            num_classes=527,
            depths=[2, 2, 6, 2],
            num_heads=[4, 8, 16, 32],
            window_size=8
        )
        
        # Load audio encoder weights directly filtering out text/projection constraints
        if os.path.exists(weights_path):
            full_state = torch.load(weights_path, map_location='cpu')
            # Extract state entries belonging strictly to the structural audio_encoder tower
            audio_state = {}
            for k, v in full_state.items():
                if k.startswith('clap.audio_encoder.base.htsat.'):
                    new_k = k.replace('clap.audio_encoder.base.htsat.', '')
                    audio_state[new_k] = v
                    
            self.htsat.load_state_dict(audio_state, strict=True)
        else:
            raise FileNotFoundError(f"Missing absolute mandatory CLAP weight asset at: {weights_path}")
            
        self.htsat.to(device)
        self.htsat.eval() # Freeze in structural prediction evaluation context
        
        for param in self.htsat.parameters():
            param.requires_grad = False

    def forward(self, raw_audio_batch, format_id, fraction_id, device='cuda'):
        """
        args:
         - raw_audio_batch: 1D audio tensors batch from CPU DataLoader [B, 358400]
         - format_id: Target audio format codec (0=WAV, 1=MP3)
         - fraction_id: Active 1/n octave denominator (1, 3, 6, 12)
         - device: Active execution device GPU
        returns:
         - x_native_norm: Pristine targets extracted using CLAP's original sub-layers [B, 1, 64, 700]
         - conditioning_C: Fused geometric spatial mask anchor [B, 1, 64, 700]
        """
        audio_signal = raw_audio_batch.to(torch.float32).to(device)
        audio_signal = torch.nan_to_num(audio_signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 1. Stochastic lossy codec simulation directly in RAM
        if format_id == 1: 
            fft_data = torch.fft.rfft(audio_signal, dim=-1)
            nyquist = self.sample_rate / 2.0
            cutoff_bin = int((16000.0 / nyquist) * fft_data.shape[-1])
            fft_data[:, cutoff_bin:] = 0.0 
            audio_signal = torch.fft.irfft(fft_data, dim=-1)
            audio_signal += torch.randn_like(audio_signal) * 1e-4

        # 2. Replicate CLAP Native Preprocessing pipeline for x_0 target
        with torch.no_grad():
            x_stft = self.htsat.spectrogram_extractor(audio_signal)
            x_native_logmel = self.htsat.logmel_extractor(x_stft)
            x_native_norm = self.htsat.bn0(x_native_logmel.transpose(1, 3)).transpose(1, 3)
        
        x_native_norm = x_native_norm.permute(0, 1, 3, 2) 

        # 3. Generate low-resolution structural condition filterbank C
        octave_spec = spectrogram_n_octaveband_generator_gpu(
            wav_batch=audio_signal,
            sampling_rate=self.sample_rate,
            n_octave=fraction_id,
            center_freqs=None,
            ref=2e-5,
            device=device
        ) #
        
        octave_spec = octave_spec.permute(0, 2, 1)
        conditioning_C = convert_octave_to_msclap_mel(octave_spec, target_mels=64) #
        conditioning_C = conditioning_C.permute(0, 1, 3, 2)
        
        if conditioning_C.shape[-1] != x_native_norm.shape[-1]:
            conditioning_C = F.interpolate(
                conditioning_C,
                size=(64, x_native_norm.shape[-1]),
                mode='bilinear',
                align_corners=False
            )

        return x_native_norm, conditioning_C

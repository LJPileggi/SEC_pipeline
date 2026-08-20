import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import huggingface_hub
import transformers
import msclap

# Dynamic root injection to safely import core production modules from src/
current_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if src_root not in sys.path: 
    sys.path.insert(0, src_root)

from src.utils import VERBOSE

INJECT_OCTAVE = os.environ.get("INJECT_OCTAVE", "False").lower() == "true"

def universal_path_redirect(*args, **kwargs):
    rank = os.environ.get('SLURM_PROCID', '0')
    weights_path = os.getenv("LOCAL_CLAP_WEIGHTS_PATH")
    text_path = os.getenv("CLAP_TEXT_ENCODER_PATH")

    if any(x for x in args if 'msclap' in str(x)) or 'CLAP_weights' in str(kwargs):
        return weights_path

    filename = kwargs.get('filename') or (args[1] if len(args) > 1 else None)
    if filename and text_path:
        forced_target = os.path.join(text_path, str(filename))
        if VERBOSE:
            print(f"🎯 [Rank {rank}] FIREWALL REDIRECT: {filename} -> {forced_target}", flush=True)
        return forced_target

    return text_path

huggingface_hub.hf_hub_download = universal_path_redirect
transformers.utils.hub.cached_file = universal_path_redirect
transformers.utils.hub.hf_hub_download = universal_path_redirect
msclap.CLAPWrapper.hf_hub_download = universal_path_redirect

if INJECT_OCTAVE:
    try:
        from msclap.models.htsat import HTSAT_N_Level
        def patched_forward(self, x):
            if isinstance(x, torch.Tensor) and x.ndim == 4:
                return self.forward_features(x)
            return self.original_forward(x)

        if not hasattr(HTSAT_N_Level, 'original_forward'):
            HTSAT_N_Level.original_forward = HTSAT_N_Level.forward
            HTSAT_N_Level.forward = patched_forward
            
        if VERBOSE:
            print("💉 MSCLAP PATCH: HTSAT_N_Level 'forward' successfully bypassed.")
    except ImportError:
        if VERBOSE:
            print("⚠️ WARNING: Could not find HTSAT_N_Level in msclap.models.htsat.")

from src.models import spectrogram_n_octaveband_generator_gpu, reshape_spectrogram, CLAP_initializer

class SpectralConvergenceLoss(nn.Module):
    """
    Spectral Convergence Loss: misura l'errore relativo della norma di Frobenius
    tra lo spettrogramma Log-Mel target pristine (x0) e quello stimato dal modello.
    Evita l'appiattimento spettrale tipico della sola MSE.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x_pred, x_target):
        diff_norm = torch.norm(x_target - x_pred, p='fro')
        target_norm = torch.norm(x_target, p='fro') + 1e-8
        return diff_norm / target_norm

class OnlineSpectrogramPipeline(nn.Module):
    def __init__(self, weights_path, sample_rate=52100, device='cuda'):
        super().__init__()
        self.sample_rate = sample_rate
        use_cuda = True if 'cuda' in str(device) else False
        
        clap_object, _, _ = CLAP_initializer(device=device, use_cuda=use_cuda)
        self.htsat = clap_object.clap.audio_encoder.base.htsat
        self.htsat.to(device)
        self.htsat.eval()
        
        for param in self.htsat.parameters():
            param.requires_grad = False

    def forward(self, raw_audio_batch, format_id, fraction_id, device='cuda'):
        audio_signal = raw_audio_batch.to(torch.float32).to(device)
        audio_signal = torch.nan_to_num(audio_signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 1. Stochastic lossy codec simulation
        if format_id == 1: 
            fft_data = torch.fft.rfft(audio_signal, dim=-1)
            nyquist = self.sample_rate / 2.0
            cutoff_bin = int((16000.0 / nyquist) * fft_data.shape[-1])
            fft_data[:, cutoff_bin:] = 0.0 
            audio_signal = torch.fft.irfft(fft_data, dim=-1)
            audio_signal += torch.randn_like(audio_signal) * 1e-4

        # 🎯 PULIZIA RIGIDA DIVERGENZE NUMERICHE PER EVITARE CUDA LAUNCH FAILURE
        audio_signal = torch.nan_to_num(audio_signal, nan=0.0, posinf=0.0, neginf=0.0)
        audio_signal = torch.clamp(audio_signal, min=-10.0, max=10.0)

        # 2. Replicate CLAP Native Preprocessing (64 mels target x_0)
        with torch.no_grad():
            x_stft = self.htsat.spectrogram_extractor(audio_signal)
            x_native_logmel = self.htsat.logmel_extractor(x_stft)
            x_native_norm = self.htsat.bn0(x_native_logmel.transpose(1, 3)).transpose(1, 3)
        
        x_native_norm = x_native_norm.permute(0, 1, 3, 2) 

        # 3. Generate 32-nd octave condition C (332 channels) in Float32
        with torch.cuda.amp.autocast(enabled=False):
            audio_signal_fp32 = audio_signal.float() 
            octave_spec = spectrogram_n_octaveband_generator_gpu(
                wav_batch=audio_signal_fp32,
                sampling_rate=self.sample_rate,
                n_octave=fraction_id,
                center_freqs=None,
                ref=2e-5,
                device=device
            )
        
        octave_spec = octave_spec.permute(0, 2, 1)
        # 🎯 Reshaping a 332 canali per le 32esime d'ottava
        conditioning_C = reshape_spectrogram(octave_spec, target_dim=332)
        conditioning_C = conditioning_C.permute(0, 1, 3, 2) # Shape: [B, 1, 332, T_blocks]
        
        if conditioning_C.shape[-1] != x_native_norm.shape[-1]:
            conditioning_C = F.interpolate(
                conditioning_C,
                size=(332, x_native_norm.shape[-1]),
                mode='bilinear',
                align_corners=False
            )

        return x_native_norm, conditioning_C

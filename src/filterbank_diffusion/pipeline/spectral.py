import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import huggingface_hub
import transformers
import msclap

current_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from src.utils import VERBOSE

INJECT_OCTAVE = os.environ.get("INJECT_OCTAVE", "False").lower() == "true"

def universal_path_redirect(*args, **kwargs):
    weights_path = os.getenv("LOCAL_CLAP_WEIGHTS_PATH")
    text_path = os.getenv("CLAP_TEXT_ENCODER_PATH")

    if any(x for x in args if 'msclap' in str(x)) or 'CLAP_weights' in str(kwargs):
        return weights_path

    filename = kwargs.get('filename') or (args[1] if len(args) > 1 else None)
    if filename and text_path:
        return os.path.join(text_path, str(filename))

    return text_path

huggingface_hub.hf_hub_download = universal_path_redirect
transformers.utils.hub.cached_file = universal_path_redirect
transformers.utils.hub.hf_hub_download = universal_path_redirect
msclap.CLAPWrapper.hf_hub_download = universal_path_redirect

from src.models import spectrogram_n_octaveband_generator_gpu, convert_octave_to_msclap_mel, CLAP_initializer

class SpectralConvergenceLoss(nn.Module):
    """Relative Frobenius norm error between predicted and target mel spectrograms."""
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

        audio_signal = torch.nan_to_num(audio_signal, nan=0.0, posinf=0.0, neginf=0.0)
        audio_signal = torch.clamp(audio_signal, min=-10.0, max=10.0)

        # 2. Extract clean target Log-Mel (x_0 pristine: [B, 1, 64, 700])
        with torch.no_grad():
            x_stft = self.htsat.spectrogram_extractor(audio_signal)
            x_native_logmel = self.htsat.logmel_extractor(x_stft)
            x_native_norm = self.htsat.bn0(x_native_logmel.transpose(1, 3)).transpose(1, 3)
        
        x_0_pristine = x_native_norm.permute(0, 1, 3, 2) # [B, 1, 64, 700]

        # 3. Generate octave-band spectrogram on GPU
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
        
        octave_spec = octave_spec.permute(0, 2, 1) # [B, T_blocks, F_octave]
        
        # 4. Joint 2D interpolation directly to [B, 1, 64, 700] with CLAP bn0 alignment
        x_cond = convert_octave_to_msclap_mel(octave_spec, target_mels=64, target_time=x_0_pristine.shape[-1])

        return x_0_pristine, x_cond

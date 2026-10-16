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

from src.models import spectrogram_n_octaveband_generator_gpu, hyperresolve_octave_pchip_gpu, CLAP_initializer


class AcousticStructuralLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x_pred, x_target):
        l_recon = F.l1_loss(x_pred, x_target)

        diff_t_pred = x_pred[..., 1:] - x_pred[..., :-1]
        diff_t_target = x_target[..., 1:] - x_target[..., :-1]
        l_transient = F.l1_loss(diff_t_pred, diff_t_target)

        diff_f_pred = x_pred[..., 1:, :] - x_pred[..., :-1, :]
        diff_f_target = x_target[..., 1:, :] - x_target[..., :-1, :]
        l_texture = F.l1_loss(diff_f_pred, diff_f_target)

        x_pred_s2 = F.avg_pool2d(x_pred, kernel_size=(2, 2))
        x_target_s2 = F.avg_pool2d(x_target, kernel_size=(2, 2))
        l_ms_s2 = F.l1_loss(x_pred_s2, x_target_s2)

        x_pred_s4 = F.avg_pool2d(x_pred, kernel_size=(4, 4))
        x_target_s4 = F.avg_pool2d(x_target, kernel_size=(4, 4))
        l_ms_s4 = F.l1_loss(x_pred_s4, x_target_s4)

        return l_recon + self.alpha * l_transient + self.beta * l_texture + 0.5 * (l_ms_s2 + l_ms_s4)


def feature_matching_loss(feats_real, feats_fake):
    loss = 0.0
    for f_r, f_f in zip(feats_real, feats_fake):
        loss += F.l1_loss(f_f, f_r.detach())
    return loss / len(feats_real)


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
        with torch.inference_mode():
            audio_signal = raw_audio_batch.to(torch.float32).to(device)
            audio_signal = torch.nan_to_num(audio_signal, nan=0.0, posinf=0.0, neginf=0.0)

            if format_id == 1:
                fft_data = torch.fft.rfft(audio_signal, dim=-1)
                nyquist = self.sample_rate / 2.0
                cutoff_bin = int((16000.0 / nyquist) * fft_data.shape[-1])
                fft_data[:, cutoff_bin:] = 0.0
                audio_signal = torch.fft.irfft(fft_data, dim=-1)
                audio_signal += torch.randn_like(audio_signal) * 1e-4
                del fft_data

            audio_signal = torch.nan_to_num(audio_signal, nan=0.0, posinf=0.0, neginf=0.0)
            audio_signal = torch.clamp(audio_signal, min=-10.0, max=10.0)

            # Mel Pristine
            x_stft = self.htsat.spectrogram_extractor(audio_signal)
            x_native_logmel = self.htsat.logmel_extractor(x_stft)
            x_native_norm = self.htsat.bn0(x_native_logmel.transpose(1, 3)).transpose(1, 3)
            del x_stft, x_native_logmel

            x_target_native = x_native_norm.permute(0, 1, 3, 2).contiguous()
            del x_native_norm

            target_time = 1152
            if x_target_native.shape[-1] < target_time:
                pad_amount = target_time - x_target_native.shape[-1]
                x_0_pristine = F.pad(x_target_native, (0, pad_amount, 0, 0), mode='constant', value=0.0)
            else:
                x_0_pristine = x_target_native[:, :, :, :target_time]
            del x_target_native

            # Octave Filterbank GPU
            audio_signal_fp32 = audio_signal.float()
            octave_spec = spectrogram_n_octaveband_generator_gpu(
                wav_batch=audio_signal_fp32,
                sampling_rate=self.sample_rate,
                n_octave=fraction_id,
                center_freqs=None,
                ref=2e-5,
                device=device
            )
            del audio_signal, audio_signal_fp32

            octave_spec = octave_spec.permute(0, 2, 1)

            x_octave_pchip = hyperresolve_octave_pchip_gpu(
                octave_spec,
                n_octave=fraction_id,
                target_bands=320,
                target_time=target_time,
                sample_rate=self.sample_rate
            )
            del octave_spec

        return x_0_pristine, x_octave_pchip

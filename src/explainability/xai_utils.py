import numpy as np
import librosa
import torch
import soundfile as sf
from pydub import AudioSegment
import os

def spectrogram_to_audio(spec_tensor, sampling_rate, n_fft=2048, n_iter=16):
    """
    Mappa lo spettrogramma n-octave in bin STFT e ricostruisce l'audio.
    Condiviso tra L-MAC e SLIME.
    """
    spec_np = spec_tensor.detach().cpu().numpy().squeeze()
    if spec_np.ndim == 1: 
        spec_np = spec_np.reshape(-1, 1)
        
    stft_approx = np.zeros((n_fft // 2 + 1, spec_np.shape[1]), dtype=np.float32)
    n_bins, n_bands = stft_approx.shape[0], spec_np.shape[0]
    hop = n_bins // n_bands
    
    for i in range(n_bands):
        start_bin = i * hop
        end_bin = min((i + 1) * hop, n_bins)
        stft_approx[start_bin:end_bin, :] = spec_np[i, :]

    recon_audio = librosa.griffinlim(stft_approx, n_iter=n_iter) 
    return torch.from_numpy(recon_audio).float().to(spec_tensor.device).unsqueeze(0)

def save_explanation_audio(audio_tensor, sampling_rate, save_path, audio_format='wav'):
    """
    Salvataggio flessibile: usa soundfile per wav e pydub per il resto.
    """
    audio_np = audio_tensor.squeeze().cpu().numpy()
    
    if audio_format.lower() == 'wav':
        sf.write(save_path, audio_np, sampling_rate)
    else:
        # Conversione temporanea per pydub (float -> int16)
        audio_int = (audio_np * 32767).astype(np.int16)
        seg = AudioSegment(
            audio_int.tobytes(), 
            frame_rate=sampling_rate,
            sample_width=2, 
            channels=1
        )
        seg.export(save_path, format=audio_format.lower())

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from src.models import FinetunedModel

class Decoder(nn.Module):
    """
    Decoder network (M_theta) that generates a binary mask M from CLAP embeddings.
    Maps a 1024D vector to the (Freq, Time) dimensions of the n-octave spectrogram.
    """
    def __init__(self, latent_dim_input=1024, output_spectrogram_shape=(27, 256)):
        super(Decoder, self).__init__()
        # 🎯 ASSICURIAMO che sia una tupla di int
        self.output_shape = tuple(int(s) for s in output_spectrogram_shape)
        
        # Initial projection to spatial feature maps
        self.fc = nn.Linear(latent_dim_input, 128 * 8 * 8)
        
        # Upsampling blocks to reach target resolution
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid() # Masks must be in [0, 1] range
        )

    def forward(self, h):
        # h shape: (batch, 1024)
        x = self.fc(h).view(-1, 128, 8, 8)
        mask = self.deconv(x)
        # Interpolate to exact n-octave band dimensions (e.g., 27 bands)
        mask = F.interpolate(mask, size=self.output_shape, mode='bilinear', align_corners=False)
        return mask

class LMAC:
    """
    Listenable Maps for Audio Classifiers (L-MAC).
    Core logic for generating and training audio interpretability masks.
    """
    def __init__(self, classifier, decoder, clap_model, audio_embedding, lambda_in=1.0, lambda_s=0.001):
        self.classifier = classifier # This is the FinetunedModel passed from the pipeline
        self.decoder = decoder
        self.clap_model = clap_model
        self.audio_embedding = audio_embedding
        self.lambda_in = lambda_in
        self.lambda_s = lambda_s
        self.ce_loss = nn.CrossEntropyLoss()

    def _apply_mask_to_audio(self, masked_spec, sampling_rate):
        spec_np = masked_spec.detach().cpu().numpy().squeeze()
        if spec_np.ndim == 1: spec_np = spec_np.reshape(-1, 1) # Safety for single frames
        
        n_fft = 2048
        stft_approx = np.zeros((n_fft // 2 + 1, spec_np.shape[1]), dtype=np.float32)
        
        # 🎯 MAPPING: Distribuiamo le 27 bande sulle frequenze STFT
        # In produzione useremo i center_freqs, qui facciamo uno stretching lineare
        n_bins = stft_approx.shape[0]
        n_bands = spec_np.shape[0]
        hop = n_bins // n_bands
        
        for i in range(n_bands):
            start_bin = i * hop
            end_bin = min((i + 1) * hop, n_bins)
            stft_approx[start_bin:end_bin, :] = spec_np[i, :] # Spalmiamo l'energia

        recon_audio = librosa.griffinlim(stft_approx, n_iter=16) 
        return torch.from_numpy(recon_audio).float().to(masked_spec.device).unsqueeze(0)

    def calculate_masking_loss(self, h_original, linear_spec_X, sampling_rate):
        """
        Computes the masking loss. Trains the decoder to keep only the features 
        that sustain the classifier's original prediction.
        """
        # 1. Get reference prediction
        with torch.no_grad():
            y_hat = torch.argmax(self.classifier(h_original), dim=1)

        # 2. Generate mask and apply to the HDF5 linear spectrogram
        M = self.decoder(h_original)
        X_masked = M * linear_spec_X

        # 3. Audio Loopback: Masked Spec -> Waveform -> New Embedding
        audio_in = self._apply_mask_to_audio(X_masked, sampling_rate)
        h_in = self.audio_embedding(audio_in)[0][0]
        
        # 4. Final classification of the "masked-in" part
        logits_in = self.classifier(h_in)
        
        # 5. Losses: Classification fidelity + Mask Sparsity
        l_in = self.ce_loss(logits_in, y_hat)
        l_s = torch.mean(torch.abs(M))
        
        return (self.lambda_in * l_in) + (self.lambda_s * l_s), M

    def generate_listenable_interpretation(self, M, linear_spec_X, sampling_rate, save_path=None):
        """
        Final output function: applies mask and generates a high-quality .wav file 
        for human inspection.
        """
        X_final = M * linear_spec_X
        waveform = self._apply_mask_to_audio(X_final, sampling_rate)
        
        if save_path:
            # Logic to save via soundfile or librosa
            pass
            
        return waveform

import torch
import torch.nn as nn
from .xai_utils import spectrogram_to_audio, save_explanation_audio

class Decoder(nn.Module):
    def __init__(self, latent_dim_input=1024, output_spectrogram_shape=(27, 256)):
        super(Decoder, self).__init__()
        self.output_shape = tuple(int(s) for s in output_spectrogram_shape)
        self.fc = nn.Linear(latent_dim_input, 128 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, h):
        x = self.fc(h)
        x = x.view(-1, 128, 8, 8)
        mask = self.deconv(x)
        # Resize dinamico per matchare esattamente lo spettrogramma lineare di HDF5
        return nn.functional.interpolate(mask, size=self.output_shape, mode='bilinear', align_corners=False)

class LMAC:
    def __init__(self, classifier, audio_embedding, decoder, lambda_in=1.0, lambda_s=0.01):
        self.classifier = classifier
        self.audio_embedding = audio_embedding
        self.decoder = decoder
        self.lambda_in = lambda_in
        self.lambda_s = lambda_s
        self.ce_loss = nn.CrossEntropyLoss()

    def compute_loss(self, h_original, linear_spec_X, sampling_rate):
        with torch.no_grad():
            y_hat = torch.argmax(self.classifier(h_original), dim=1)

        M = self.decoder(h_original)
        X_masked = M * linear_spec_X
        
        # Uso della utility condivisa
        audio_in = spectrogram_to_audio(X_masked, sampling_rate)
        
        output = self.audio_embedding(audio_in)
        h_in = output[0][0] if isinstance(output, (tuple, list)) else output
        
        logits_in = self.classifier(h_in)
        l_in = self.ce_loss(logits_in, y_hat)
        l_s = torch.mean(torch.abs(M))
        
        return (self.lambda_in * l_in) + (self.lambda_s * l_s), M

    def generate_listenable_interpretation(self, M, linear_spec_X, sampling_rate, save_path=None, audio_format='wav'):
        """
        Genera l'audio e lo salva (se il path è fornito) usando la logica flessibile.
        """
        X_final = M * linear_spec_X
        listenable_audio = spectrogram_to_audio(X_final, sampling_rate)
        
        if save_path:
            save_explanation_audio(listenable_audio, sampling_rate, save_path, audio_format)
            
        return listenable_audio

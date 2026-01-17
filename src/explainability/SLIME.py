import torch
import numpy as np
from sklearn.linear_model import Ridge
from .xai_utils import spectrogram_to_audio

class SLIME:
    """
    Sound LIME (SLIME) for Music Content Analysis.
    Learns local surrogate models over interpretable segments. [cite: 8, 46, 100]
    """
    def __init__(self, classifier, audio_embedding, explainer_type='time_frequency', n_samples=1000):
        self.classifier = classifier
        self.audio_embedding = audio_embedding
        self.explainer_type = explainer_type
        self.n_samples = n_samples

    def _generate_interpretable_components(self, audio_waveform, spec_linear):
        """
        Segments instance into interpretable components Xi. [cite: 105, 157, 158]
        """
        if self.explainer_type == 'time':
            # Temporal partitioning into super samples [cite: 157]
            # Assuming 10 segments for 1 second of audio as per paper [cite: 203]
            n_segments = 10
            seg_len = len(audio_waveform) // n_segments
            components = [audio_waveform[i*seg_len : (i+1)*seg_len] for i in range(n_segments)]
            return components, [f"T{i}" for i in range(n_segments)]

        elif self.explainer_type == 'time_frequency':
            # Time-frequency partitioning into blocks Bi 
            spec_np = spec_linear.detach().cpu().numpy().squeeze()
            n_f_seg, n_t_seg = 4, 6 # Common paper configuration [cite: 151, 263]
            f_step = spec_np.shape[0] // n_f_seg
            t_step = spec_np.shape[1] // n_t_seg
            
            names = [f"B{i}" for i in range(n_f_seg * n_t_seg)]
            return (n_f_seg, n_t_seg, f_step, t_step), names

    def explain_instance(self, audio_waveform, spec_linear, sampling_rate, class_idx):
        """
        Learns the linear explanation model g over interpretable space {0,1}. [cite: 106, 109, 165]
        """
        components, names = self._generate_interpretable_components(audio_waveform, spec_linear)
        n_comp = len(names)
        
        # 1. Generate N synthetic samples in binary space [cite: 107, 165, 204]
        synthetic_interpretable = np.random.randint(0, 2, size=(self.n_samples, n_comp))
        synthetic_interpretable[0, :] = 1 # Include original instance [cite: 165]

        # 2. Get classifier predictions for perturbed samples [cite: 119, 168]
        predictions = []
        for i in range(self.n_samples):
            # Apply mask based on explainer type
            masked_audio = self._apply_mask(audio_waveform, spec_linear, components, 
                                            synthetic_interpretable[i], sampling_rate)
            
            with torch.no_grad():
                # Loopback via CLAP e Classifier
                output = self.audio_embedding(masked_audio)
                h_in = output[0][0] if isinstance(output, (tuple, list)) else output

                # 🎯 DEBUG: Vediamo se l'embedding cambia!
                if i < 3:
                    print(f"DEBUG Sample {i} - Embedding Mean: {h_in.mean().item():.8f}, Std: {h_in.std().item():.8f}")
                
                # Assicuriamoci che h_in abbia la dimensione del batch per il classificatore
                if h_in.dim() == 1:
                    h_in = h_in.unsqueeze(0)
                
                logits = self.classifier(h_in)
                probs = torch.softmax(logits, dim=-1)
                
                # 🎯 CORREZIONE: Gestione flessibile della forma del tensore
                # Se probs è [1, N], prendiamo [0, class_idx]. Se è [N], prendiamo [class_idx].
                if probs.dim() > 1:
                    val = probs[0, class_idx].item()
                else:
                    val = probs[class_idx].item()
                
                predictions.append(val)

        # 3. Kernel weighting based on distance to original instance [cite: 152, 168, 171]
        distances = np.linalg.norm(synthetic_interpretable - 1, axis=1)
        weights = np.exp(-distances**2 / (0.25**2))

        # 4. Train local linear regressor (Ridge) [cite: 108, 169]
        regressor = Ridge(alpha=1.0)
        regressor.fit(synthetic_interpretable, predictions, sample_weight=weights)
        
        return {names[i]: regressor.coef_[i] for i in range(n_comp)}

    def _apply_mask(self, audio_waveform, spec_linear, components, mask_vec, sampling_rate):
        """Applies binary mask to interpretable components. [cite: 107, 167]"""
        if self.explainer_type == 'time':
            reconstructed = torch.zeros_like(audio_waveform)
            seg_len = len(audio_waveform) // len(mask_vec)
            for i, val in enumerate(mask_vec):
                if val == 1:
                    reconstructed[i*seg_len : (i+1)*seg_len] = audio_waveform[i*seg_len : (i+1)*seg_len]
            return reconstructed.unsqueeze(0)
        
        else: # time_frequency
            n_f_seg, n_t_seg, f_step, t_step = components
            spec_masked = spec_linear.clone().squeeze()
            for idx, val in enumerate(mask_vec):
                if val == 0:
                    i, j = divmod(idx, n_t_seg)
                    spec_masked[i*f_step:(i+1)*f_step, j*t_step:(j+1)*t_step] = 0
            return spectrogram_to_audio(spec_masked, sampling_rate)

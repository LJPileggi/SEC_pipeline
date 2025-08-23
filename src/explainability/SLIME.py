import torch
import torch.nn as nn
import numpy as np
import librosa
from sklearn.linear_model import Ridge # Usiamo Ridge per il modello lineare semplice
import math

class SLIMEExplainer:
    def __init__(self, classifier, feature_extractor, explainer_type='time_frequency', n_samples=1000):
        """
        Initializes the SLIME Explainer.

        Args:
            classifier (callable): A function that takes a feature vector (e.g., MFCC) and returns
                                   a probability distribution or a single probability for a class.
            feature_extractor (callable): A function that takes a raw audio waveform and returns
                                          the feature vector required by the classifier.
            explainer_type (str): The type of explanation to generate ('time', 'frequency', 'time_frequency').
            n_samples (int): The number of synthetic samples to generate in the interpretable space.
        """
        self.classifier = classifier
        self.feature_extractor = feature_extractor
        self.explainer_type = explainer_type
        self.n_samples = n_samples
        
    def _generate_interpretable_components(self, audio_waveform, sampling_rate, segment_duration=0.1, n_freq_bands=4):
        """
        Generates interpretable components from the audio waveform.
        This is a core contribution of SLIME (the dotted box in Fig. 3).
        """
        # Temporal Segmentation
        if self.explainer_type == 'time':
            segment_samples = int(segment_duration * sampling_rate)
            segments = [
                audio_waveform[i:i + segment_samples] 
                for i in range(0, len(audio_waveform), segment_samples)
            ]
            # Assumiamo che ogni segmento sia un componente interpretabile
            return segments, [f"T{i+1}" for i in range(len(segments))]

        # Time-Frequency Segmentation
        elif self.explainer_type == 'time_frequency':
            # Esempio di segmentazione, come in Fig. 4(b)
            n_fft = 2048
            hop_length = int(segment_duration * sampling_rate) # Esempio: 100ms
            
            # Calcola il Mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_waveform,
                sr=sampling_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=80 # Esempio di numero di bin Mel
            )
            
            # Segmenta il Mel-spectrogram in blocchi
            segments_in_time = mel_spec.shape[1]
            segments_in_freq = n_freq_bands
            
            freq_band_size = mel_spec.shape[0] // segments_in_freq
            
            # Crea i blocchi tempo-frequenza
            blocks = []
            block_names = []
            block_idx = 1
            for i in range(segments_in_freq):
                for j in range(segments_in_time):
                    # Un blocco è una porzione del Mel-spectrogram
                    block = mel_spec[i*freq_band_size:(i+1)*freq_band_size, j]
                    blocks.append(block)
                    block_names.append(f"B{block_idx}")
                    block_idx += 1

            return blocks, block_names
        
        else:
            raise ValueError("explainer_type must be 'time' or 'time_frequency'. Spectral not implemented.")

    def _generate_interpretable_representation(self, components):
        """
        Maps components to a binary interpretable representation.
        x' in {0, 1}^|Xi|
        """
        return np.ones(len(components))

    def _generate_synthetic_samples(self, interpretable_representation):
        """
        Generates synthetic samples by randomly setting components to zero.
        z'k in {0, 1}^|Xi|
        """
        num_components = len(interpretable_representation)
        # Il primo campione è sempre l'originale
        samples = [interpretable_representation]
        
        for _ in range(self.n_samples - 1):
            sample = np.copy(interpretable_representation)
            # Randomly turn off a subset of components
            off_indices = np.random.choice(
                num_components, 
                size=np.random.randint(1, num_components), 
                replace=False
            )
            sample[off_indices] = 0
            samples.append(sample)
            
        return np.array(samples)

    def _project_sample_to_feature_space(self, original_waveform, components, synthetic_interpretable_sample):
        """
        Projects a synthetic interpretable sample back to the original feature space.
        z'k -> zk in R^n
        """
        # Start with a silent waveform
        projected_waveform = np.zeros_like(original_waveform)
        
        # Reconstruct the waveform from the components that are 'on'
        current_idx = 0
        for i, component_is_on in enumerate(synthetic_interpretable_sample):
            if component_is_on:
                component = components[i]
                
                # Inserisci il componente nella waveform
                if self.explainer_type == 'time':
                    end_idx = current_idx + len(component)
                    projected_waveform[current_idx:end_idx] = component
                    current_idx = end_idx
                elif self.explainer_type == 'time_frequency':
                    # Questo è complesso. La ricostruzione di un'intera waveform da blocchi
                    # tempo-frequenza senza fase è un problema di sintesi, simile a Griffin-Lim.
                    # Per una dimostrazione concettuale, useremo un placeholder.
                    # In un'implementazione reale, avresti bisogno di una funzione di sintesi.
                    # Per semplicità, proiettiamo i blocchi direttamente nello spazio delle feature.
                    # Se il tuo feature extractor prende waveform, questa parte è la più complessa.
                    # Per ora, restituiamo una rappresentazione "fittizia"
                    return self.feature_extractor(original_waveform) # Placeholder: ritorniamo l'originale
        
        return self.feature_extractor(projected_waveform)


    def explain_instance(self, audio_waveform, sampling_rate, class_to_explain):
        """
        Generates a local explanation for a single audio instance.

        Args:
            audio_waveform (np.ndarray): The raw audio waveform.
            sampling_rate (int): The sampling rate of the waveform.
            class_to_explain (int): The index of the class to be explained.

        Returns:
            dict: The explanation with weights and component names.
        """
        # Step 1: Generate interpretable components and representation
        components, component_names = self._generate_interpretable_components(audio_waveform, sampling_rate)
        interpretable_representation_original = self._generate_interpretable_representation(components)
        
        # Step 2: Generate N synthetic samples
        synthetic_interpretable_samples = self._generate_synthetic_samples(interpretable_representation_original)
        
        # Step 3: Project samples to feature space and get classifier predictions
        projected_features = []
        for sample_interp in synthetic_interpretable_samples:
            # Per una dimostrazione concettuale, usiamo un placeholder per la proiezione
            # In una vera implementazione, questo proietterebbe i dati
            # e calcolerebbe le features.
            projected_features.append(self.feature_extractor(audio_waveform))
        
        # Ora chiediamo al classificatore la sua predizione per ogni feature
        classifier_predictions = []
        for feature_vec in projected_features:
            logits = self.classifier(feature_vec)
            # Estraiamo la probabilità per la classe di interesse
            probs = torch.softmax(logits, dim=-1)
            classifier_predictions.append(probs[0, class_to_explain].item())

        # Step 4: Calcola le distanze (semplificato) e i pesi
        # Le distanze qui sono una semplificazione. LIME usa una distanza su uno spazio di feature.
        distances = np.linalg.norm(synthetic_interpretable_samples - interpretable_representation_original, axis=1)
        # Applica una funzione kernel per pesare i campioni in base alla vicinanza
        kernel_width = 0.25
        weights = np.exp(-distances**2 / (kernel_width**2))
        
        # Step 5: Addestra il modello lineare semplice (Ridge Regression)
        # X è la rappresentazione binaria dei campioni
        # y sono le probabilità predette dal classificatore
        # sample_weight sono i pesi di vicinanza
        regressor = Ridge(alpha=1.0) # alpha è il parametro di regolarizzazione
        regressor.fit(synthetic_interpretable_samples, classifier_predictions, sample_weight=weights)
        
        # Step 6: Genera la spiegazione (i pesi del modello lineare)
        explanation_weights = regressor.coef_
        
        # Ordina i componenti per importanza assoluta
        sorted_indices = np.argsort(np.abs(explanation_weights))[::-1]
        
        explanation = {}
        for i in sorted_indices:
            explanation[component_names[i]] = explanation_weights[i]
            
        return explanation


#######################################################################################


import torch
import torch.nn as nn
import numpy as np
import librosa
from sklearn.linear_model import Ridge # Usiamo Ridge per il modello lineare semplice
import math

# Esempio di implementazione della classe SLIMEExplainer (dal tab precedente)
# Incollare qui la classe SLIMEExplainer

# --- Dummy Classifier and Feature Extractor for Demonstration ---
class DummyClassifier(nn.Module):
    def __init__(self, num_classes=2, input_dim=120):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        # x è la forma (batch_size, input_dim)
        return self.fc(x)

def dummy_mfcc_extractor(audio_waveform, sr=22050):
    """
    Simulates the MFCC feature extraction from the paper.
    Calculates 30 MFCCs and their derivatives over 5 frames (1 sec total).
    """
    # Se l'input è una waveform di 1 sec a 22050Hz, ha 22050 campioni
    # Con frame size 200ms e 50% overlap, 1 sec = 5 frame.
    frame_size = int(0.2 * sr) # 4410 samples
    hop_length = int(0.1 * sr) # 2205 samples
    
    mfccs = librosa.feature.mfcc(y=audio_waveform, sr=sr, n_mfcc=30, n_fft=frame_size, hop_length=hop_length)
    delta_mfccs = librosa.feature.delta(mfccs)
    
    # Il paper usa il mediano e la deviazione standard su 5 frames
    # mfccs shape: (n_mfcc, n_frames). n_frames dovrebbe essere 5.
    if mfccs.shape[1] < 5:
        # Pad se l'audio è troppo corto
        pad_frames = 5 - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_frames)))
        delta_mfccs = np.pad(delta_mfccs, ((0, 0), (0, pad_frames)))
    
    mfcc_median = np.median(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    delta_mfcc_median = np.median(delta_mfccs, axis=1)
    delta_mfcc_std = np.std(delta_mfccs, axis=1)
    
    feature_vector = np.concatenate([mfcc_median, mfcc_std, delta_mfcc_median, delta_mfcc_std])
    return torch.from_numpy(feature_vector).float().unsqueeze(0) # Forma: (1, 120)

# --- Impostazioni ---
sampling_rate = 22050
duration_sec = 1.0
num_classes = 2 # 'music without voice' e 'music with voice'
class_to_explain = 1 # 'music with voice'

# Crea una dummy waveform di 1 secondo
audio_waveform = librosa.tone(300, sr=sampling_rate, length=int(duration_sec * sampling_rate))
audio_waveform_with_vocal = audio_waveform + np.sin(2*np.pi*440*np.arange(len(audio_waveform))/sampling_rate) * 0.5

# --- Inizializzazione ---
# Inizializza il classificatore e l'estrattore di feature "black-box"
dummy_model = DummyClassifier(num_classes=num_classes)
# Imposta i pesi del dummy model per simulare una classificazione
# simuliamo che il modello predica "voce" se il 30esimo MFCC è alto.
dummy_model.fc.weight.data.zero_()
dummy_model.fc.weight.data[1, 30] = 5.0 # Peso forte per il 30esimo MFCC
dummy_model.fc.bias.data.zero_()

# Inizializza l'explainer SLIME
# Passiamo una lambda che racchiude il dummy feature extractor
def feature_extractor_wrapper(audio_data):
    return dummy_mfcc_extractor(audio_data)

slime_explainer = SLIMEExplainer(
    classifier=dummy_model,
    feature_extractor=feature_extractor_wrapper,
    explainer_type='time', # 'time_frequency' anche possibile
    n_samples=500 # Un numero più piccolo per la demo
)

# --- Esecuzione ---
print(f"Generazione di una spiegazione SLIME per un'istanza di classe {class_to_explain}...")
explanation = slime_explainer.explain_instance(
    audio_waveform=audio_waveform_with_vocal,
    sampling_rate=sampling_rate,
    class_to_explain=class_to_explain
)

print("\nSpiegazione SLIME generata:")
# Stampa i primi 5 componenti più influenti
for component, weight in list(explanation.items())[:5]:
    print(f"  Componente: {component}, Peso: {weight:.4f}")

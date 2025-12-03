import os
import torch
import numpy as np
import scipy
from msclap import CLAP


### CLAP models and classifiers ###

def CLAP_initializer(device='cpu', use_cuda=False):
    """
    CLAP model initialiser
    
    returns:
     - clap_model: configured CLAP object;
     - audio_embedding: CLAP audio encoder;
     - original_parameters: CLAP original parameters.
    """
    # 1. Recupera il percorso dei pesi (dal disco locale del nodo /tmp)
    clap_weights_path = os.getenv("LOCAL_CLAP_WEIGHTS_PATH")
    
    # 2. Recupera il percorso del Text Encoder (dall'interno del Container)
    #    NOTA: Il nome corretto della variabile d'ambiente è CLAP_TEXT_ENCODER_PATH
    text_encoder_path = os.getenv("CLAP_TEXT_ENCODER_PATH")
    
    # --- Verifica dei percorsi (Solo i pesi devono esistere sulla macchina) ---
    if not clap_weights_path:
        raise ValueError("Variabile d'ambiente LOCAL_CLAP_WEIGHTS_PATH non impostata.")
    if not os.path.exists(clap_weights_path):
         raise FileNotFoundError(f"Impossibile trovare i pesi CLAP al percorso: {clap_weights_path} (Errore: il file non è stato copiato correttamente su /tmp)")

    if not text_encoder_path:
        raise ValueError("Variabile d'ambiente CLAP_TEXT_ENCODER_PATH non impostata. (Dovrebbe puntare all'interno del container)")
        
    # AGGIUNTA NECESSARIA per allinearsi con il test:
    if not os.path.exists(text_encoder_path):
         # Il messaggio è allineato a quello atteso dal test
         raise FileNotFoundError(f"Impossibile trovare l'encoder testuale CLAP a: {text_encoder_path}")

    # --- Inizializzazione CLAP (USA IL PERCORSO LOCALE) ---
    clap_model = CLAP(version='2023', use_cuda=use_cuda, pretrained_path=clap_weights_path,
                              text_path=text_encoder_path, download_if_missing=False)    
    # ... (resto del codice CLAP, non modificato)
    original_parameters = clap_model.clap.audio_encoder.to('cpu').state_dict()
    clap_model.clap.audio_encoder = clap_model.clap.audio_encoder.to(device)
    audio_embedding=clap_model.clap.audio_encoder
    for param in audio_embedding.parameters():
        param.requires_grad = False
    return clap_model, audio_embedding, original_parameters

def spectrogram_n_octaveband_generator(
        wav_data: np.array,
        sampling_rate: int,
        n_octave: int = 3,
        integration_seconds: float = 0.1,
        order: int = 4,
        center_freqs: np.array = None,
        ref: float = 2e-5
) -> np.array:
    """
    Spectrogram generator for 1/n-octave band.

    args:
     - wav_data (np.array): input audio data;
     - sampling_rate (int): sampling rate of the input audio data;
     - n_octave (int): specifies 1/n octave band (e.g., 3 for 1/3 octave, 1 for 1 octave);
     - integration_seconds (float): integration time in seconds;
     - order (int): order of the bandpass filter;
     - center_freqs (np.array): center frequencies of the 1/n-octave bands;
     - ref (float): reference value for the spectrogram.

    returns:
     - np.array: 1/n-octave band spectrogram.
    """
    # --- GESTIONE E CALCOLO DELLE FREQUENZE CENTRALI (se center_freqs è None) ---
    if center_freqs is None:
        # Frequenza centrale di riferimento (es. 1000 Hz)
        f_ref = 1000.0
        # Frequenza minima per audio processing standard
        f_min = 20.0
        # Limite superiore: frequenza di Nyquist (metà del sampling rate)
        f_max = sampling_rate / 2.0
        
        # Calcoliamo gli indici 'n' necessari per coprire il range (secondo ISO 266: fc = 1000 * 2^(n/N))
        
        # Troviamo l'indice 'n' minimo e massimo
        n_min = int(np.round(n_octave * np.log2(f_min / f_ref)))
        n_max = int(np.round(n_octave * np.log2(f_max / f_ref)))
        
        # Generiamo gli indici e calcoliamo le frequenze centrali
        n_indices = np.arange(n_min, n_max + 1)
        center_freqs = f_ref * (2**(n_indices / n_octave))
        
        # Filtriamo le frequenze che superano il limite di Nyquist
        center_freqs = center_freqs[center_freqs <= f_max]
        
        # Se non si genera nessuna frequenza (caso limite), usiamo solo la ref.
        if len(center_freqs) == 0:
            center_freqs = np.array([f_ref]) 

    # --- FINE LOGICA center_freqs ---
    
    # octave band factor
    factor = 2 ** (1 / (2 * n_octave))
    freq_d = center_freqs / factor
    freq_u = center_freqs * factor

    bands = [
        scipy.signal.butter(
            N=order,
            Wn=np.array([lower, upper]) / (sampling_rate / 2),
            btype='bandpass',
            analog=False,
            output='sos'
        ) for (lower, upper) in zip(freq_d, freq_u)
    ]

    window = int(sampling_rate * integration_seconds)
    filtered = np.array([scipy.signal.sosfilt(band, wav_data) for band in bands])

    # handling of signals shorter than integration window
    if filtered.shape[1] < window:
        spectrogram = np.zeros((filtered.shape[0], 1, window))
    else:
        filtered = filtered[:, :window * (filtered.shape[1] // window)]
        spectrogram = filtered.reshape(filtered.shape[0], -1, window)

    rms = np.sqrt(np.mean(spectrogram ** 2, axis=-1))
    
    # handling of rms == 0 to avoid log(0)
    rms[rms == 0] = np.finfo(float).eps

    spectrogram = 20 * np.log10(rms / ref)
    return spectrogram.T

class OriginalModel:
    """
    Original CLAP classifier exploiting cosine similarity between
    audio and text embeddings.
    """
    def __init__(self, classes, get_text_embeddings, device='cpu'):
        text_embeddings = get_text_embeddings([f'this is the sound of {class_}' for class_ in classes])
        with torch.no_grad():
          embed_classes = [text_embeddings[i] for i in range(len(classes))]
        self.te = torch.stack(embed_classes).to(device)
        self.device = device

    def __call__(self, x):
        sims = [[torch.nn.functional.cosine_similarity(v, t, dim=0).item() for t in self.te] for v in x]
        return torch.FloatTensor(sims).to(self.device)


class FinetunedModel(torch.nn.Module):
    """
    1024-unit classifier layer built on top of the CLAP embeddings.
    """
    def __init__(self, classes, device='cpu'):
        super().__init__()
        self.classifier = torch.nn.Linear(1024, len(classes)).to(device)
        self.classes = classes
        self.device = device
        
    def forward(self, x):
        if isinstance(x, np.ndarray):
            # Convert to torch tensor and explicitly cast to the same dtype as the model's parameters
            x = torch.from_numpy(x).to(self.classifier.weight.dtype).to(self.classifier.weight.device)
        # If input is already a tensor, ensure it's on the correct device
        elif isinstance(x, torch.Tensor):
            x = x.to(self.device)
        # Add a check to ensure dtype is consistent before the linear layer call
        if x.dtype != self.classifier.weight.dtype:
             # This case should ideally not happen with the fix above, but good for debugging
             print(f"Warning: Input tensor dtype ({x.dtype}) mismatch with model weight dtype ({self.classifier.weight.dtype})")
             x = x.to(self.classifier.weight.dtype)

        # Checks input dimension
        # If x has more dimensions than it should (es. [batch, 1, latent_dim] instead of [batch, latent_dim])
        # a squeeze might be needed.
        if x.dim() > 2: # E.g., if it's (batch_size, 1, 1024)
            x = x.squeeze(1) # Remove unitary dimension if present

        return self.classifier(x)


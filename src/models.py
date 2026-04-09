import os
import torch
import torch.nn.functional as F
import librosa
import numpy as np
import scipy

from .losses import build_optimizer, get_scores

def CLAP_initializer(device='cpu', use_cuda=False):
    import transformers
    from msclap import CLAP
    import types
    
    text_path = os.getenv("CLAP_TEXT_ENCODER_PATH") 
    rank = os.environ.get('SLURM_PROCID', '0')
    inject_octave = os.environ.get("INJECT_OCTAVE", "False").lower() == "true"
    verbose = os.environ.get("VERBOSE", "False").lower() == "true"

    # Original methods preserved as per ground truth
    original_model_from_pretrained = transformers.AutoModel.from_pretrained
    original_tokenizer_from_pretrained = transformers.AutoTokenizer.from_pretrained

    def forced_model_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        return original_model_from_pretrained(text_path, *args, **kwargs)

    def forced_tokenizer_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        kwargs['local_files_only'] = True
        return original_tokenizer_from_pretrained(text_path, *args, **kwargs)

    transformers.AutoModel.from_pretrained = forced_model_from_pretrained
    transformers.AutoTokenizer.from_pretrained = forced_tokenizer_from_pretrained

    # Model initialization
    clap_model = CLAP(version='2023', use_cuda=use_cuda)

    # 💉 NEW PATCH: AUDIO ENCODER INJECTION ON INSTANCE
    # if inject_octave:
        # try:
            # target_instance = clap_model.clap.audio_encoder.base.htsat
            
            # def patched_forward(self, x):
                # """
                # Monkey patch for HTSAT_Swin_Transformer.
                # If input is 4D (Mel), skip STFT and Log-Mel extraction.
                # """
                # if torch.is_tensor(x) and x.ndim == 4:
                    # return self.forward_features(x)
                # return self.original_forward(x)

            # if not hasattr(target_instance, 'original_forward'):
                # target_instance.original_forward = target_instance.forward
                # target_instance.forward = types.MethodType(patched_forward, target_instance)
                
            # if verbose:
                # print(f"🎯 [RANK {rank}] Patch INJECT_OCTAVE applied to clap_model.clap.audio_encoder.base.htsat", flush=True)
        # except AttributeError as e:
            # if verbose:
                # print(f"⚠️ [RANK {rank}] Patch failed: {e}", flush=True)
                
        # except AttributeError as e:
            # if verbose:
                # print(f"⚠️ [RANK {rank}] Patch failed: could not traverse the model hierarchy. Error: {e}", flush=True)

    # Clean up AutoModel/AutoTokenizer overrides
    transformers.AutoModel.from_pretrained = original_model_from_pretrained
    transformers.AutoTokenizer.from_pretrained = original_tokenizer_from_pretrained

    audio_embedding = clap_model.get_audio_embeddings
    return clap_model, audio_embedding, text_path

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
    if center_freqs is None:
        f_ref, f_min, f_max = 1000.0, 20.0, sampling_rate / 2.0
        n_min = int(np.round(n_octave * np.log2(f_min / f_ref)))
        n_max = int(np.round(n_octave * np.log2(f_max / f_ref)))
        n_indices = np.arange(n_min, n_max + 1)
        center_freqs = f_ref * (2**(n_indices / n_octave))
        # 🎯 FIX 1: Filtro preventivo bande eccessive
        center_freqs = center_freqs[center_freqs < f_max - 50.0]

    # octave band factor
    factor = 2 ** (1 / (2 * n_octave))
    freq_d = center_freqs / factor
    freq_u = center_freqs * factor

    # 🎯 FIX 2: Clamping di sicurezza per Scipy (Wn deve essere 0 < Wn < 1)
    nyquist = sampling_rate / 2.0
    freq_d = np.clip(freq_d, 1.0, nyquist - 1.0)
    freq_u = np.clip(freq_u, 2.0, nyquist - 1.0)

    bands = [
        scipy.signal.butter(
            N=order,
            Wn=np.array([lower, upper]) / nyquist,
            btype='bandpass',
            analog=False,
            output='sos'
        ) for (lower, upper) in zip(freq_d, freq_u)
    ]

    window = int(sampling_rate * integration_seconds)
    filtered = np.array([scipy.signal.sosfilt(band, wav_data) for band in bands])

    if filtered.shape[1] < window:
        spectrogram = np.zeros((filtered.shape[0], 1, window))
    else:
        filtered = filtered[:, :window * (filtered.shape[1] // window)]
        spectrogram = filtered.reshape(filtered.shape[0], -1, window)

    rms = np.sqrt(np.mean(spectrogram ** 2, axis=-1))
    rms[rms == 0] = np.finfo(float).eps

    spectrogram = 20 * np.log10(rms / ref)
    return spectrogram.T

def spectrogram_n_octaveband_generator_gpu(wav_batch, sampling_rate, n_octave=3, center_freqs=None, ref=2e-5, device='cuda'):
    """
    Spectrogram generator for 1/n-octave band vectorised for GPU usage.

    args:
     - wav_batch (np.array): batch of input audio data;
     - sampling_rate (int): sampling rate of the input audio data;
     - n_octave (int): specifies 1/n octave band (e.g., 3 for 1/3 octave, 1 for 1 octave);
     - center_freqs (np.array): center frequencies of the 1/n-octave bands;
     - ref (float): reference value for the spectrogram;
     - device (str): device to run generation on (default 'cuda').

    returns:
     - np.array: 1/n-octave band spectrogram batch.
    """
    import torchaudio.functional as F
    
    wav_batch = wav_batch.to(torch.float32) 
    if wav_batch.dim() == 1:
        wav_batch = wav_batch.unsqueeze(0)
    
    batch_size = wav_batch.shape[0]
    nyquist = sampling_rate / 2.0
    
    if center_freqs is None:
        f_ref, f_min, f_max = 1000.0, 20.0, nyquist
        n_min = int(np.round(n_octave * np.log2(f_min / f_ref)))
        n_max = int(np.round(n_octave * np.log2(f_max / f_ref)))
        n_indices = torch.arange(n_min, n_max + 1, device=device)
        center_freqs = f_ref * (2**(n_indices / n_octave))
        center_freqs = center_freqs[center_freqs < f_max - 100.0]

    n_bands = len(center_freqs)
    factor = 2 ** (1 / (2 * n_octave))
    freq_d, freq_u = center_freqs / factor, center_freqs * factor
    
    low_np = np.clip(freq_d.cpu().numpy(), 1.0, nyquist - 1.0)
    high_np = np.clip(freq_u.cpu().numpy(), 2.0, nyquist - 1.0)
    
    sos_coeffs = []
    for lower, upper in zip(low_np, high_np):
        sos = scipy.signal.butter(N=4, Wn=np.array([lower, upper]) / nyquist, 
                                 btype='bandpass', output='sos')
        sos_coeffs.append(torch.from_numpy(sos).float())
    
    sos_tensor = torch.stack(sos_coeffs).to(device).to(torch.float32)
    
    # 🎯 MEMORY MANAGEMENT: Define micro-batch size for filtering
    # 1024 is a safe compromise between speed and memory for 10s segments
    MICRO_BATCH_SIZE = 1024 
    full_expanded_size = batch_size * n_bands
    
    # Repeat audio for each band
    x_full = wav_batch.repeat_interleave(n_bands, dim=0)
    filtered_results = []

    # 🎯 CHUNKED FILTERING LOOP
    # We process subsets of (batch_sample * band) combinations
    for i in range(0, full_expanded_size, MICRO_BATCH_SIZE):
        end_i = min(i + MICRO_BATCH_SIZE, full_expanded_size)
        x_chunk = x_full[i:end_i]
        
        # Get the corresponding SOS coefficients for the bands in this chunk
        # Each sample in wav_batch is repeated n_bands times
        band_indices = torch.arange(i, end_i, device=device) % n_bands
        sos_chunk = sos_tensor[band_indices]

        # Apply the 6-step SOS filtering (4th order = 2 sections)
        # Note: Scipy output='sos' returns [N_sections, 6]
        for s in range(sos_tensor.shape[1]):
            b = sos_chunk[:, s, :3]
            a = sos_chunk[:, s, 3:]
            x_chunk = F.lfilter(x_chunk, a, b, clamp=True)
        
        filtered_results.append(x_chunk)
        
    # Reassemble filtered signals
    x = torch.cat(filtered_results, dim=0)
    
    filtered = x.reshape(batch_size, n_bands, -1)
    
    window = int(sampling_rate * 0.1)
    n_windows = filtered.shape[2] // window
    filtered = filtered[:, :, :n_windows*window].reshape(batch_size, n_bands, n_windows, window)
    
    rms = torch.sqrt(torch.mean(filtered**2, dim=-1))
    rms = torch.clamp(rms, min=torch.finfo(torch.float32).eps)
    
    res = 20 * torch.log10(rms / ref).permute(0, 2, 1)
    return torch.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)

def spectrogram_to_audio_batch(spec_tensor, sampling_rate, n_fft=2048, n_iter=16):
    """
    Reconstructs a batch of audio signals from n-octave band spectrograms.
    This function bridges the gap between octave-based energy representations 
    and the time-domain signals expected by pre-trained encoders like CLAP.
    
    Args:
        spec_tensor (torch.Tensor): Input batch [Batch, Time, Bands].
        sampling_rate (int): Output audio sampling rate.
        n_fft (int): STFT window size for reconstruction.
        n_iter (int): Number of Griffin-Lim iterations for phase estimation.
        
    Returns:
        torch.Tensor: Reconstructed batch [Batch, Samples] on the same device.
    """
    device = spec_tensor.device
    
    # 1. Prepare data for Librosa (expects Batch x Freq x Time)
    # Transposing from [B, T, F_oct] to [B, F_oct, T]
    spec_np = spec_tensor.detach().cpu().numpy().transpose(0, 2, 1)
    
    batch_size, n_bands, n_frames = spec_np.shape
    n_bins = n_fft // 2 + 1
    
    # 2. Linear Spectral Mapping (Approximate Inverse Filter Bank)
    # Initialize an empty STFT magnitude matrix: [Batch, Freq_STFT, Time]
    stft_approx = np.zeros((batch_size, n_bins, n_frames), dtype=np.float32)
    
    # Map octave bands to STFT bins using a Zero-Order Hold approach
    # This preserves the Power Spectral Density (PSD) across the linear grid
    hop = n_bins // n_bands
    for i in range(n_bands):
        start_bin = i * hop
        end_bin = min((i + 1) * hop, n_bins)
        # Vectorized assignment across the entire batch
        stft_approx[:, start_bin:end_bin, :] = spec_np[:, i:i+1, :]

    # 3. Phase Estimation via Griffin-Lim
    # Librosa's griffinlim handles 3D tensors by processing each batch element
    # Hop length is set to standard n_fft // 4 to ensure enough overlap for phase consistency
    recon_audio = librosa.griffinlim(
        stft_approx, 
        n_iter=n_iter, 
        hop_length=n_fft // 4,
        momentum=0.99 # Faster convergence for standard acoustic signals
    )
    
    # 4. Final conversion to Tensor [Batch, Samples]
    return torch.from_numpy(recon_audio).float().to(device)

def get_octave_to_mel_transition_matrix(n_octave, n_mels=64, sample_rate=52000, device='cuda'):
    """
    Pre-computes the transition matrix W [n_bands x n_mels] for energy projection.
    """
    nyquist = sample_rate / 2.0
    f_ref, f_min, f_max = 1000.0, 20.0, nyquist
    
    # Octave center frequencies calculation (matches the generator logic)
    n_min = int(np.round(n_octave * np.log2(f_min / f_ref)))
    n_max = int(np.round(n_octave * np.log2(f_max / f_ref)))
    n_indices = np.arange(n_min, n_max + 1)
    c_freqs = f_ref * (2**(n_indices / n_octave))
    c_freqs = c_freqs[c_freqs < f_max - 50.0]
    
    # Mel filterbank generation
    n_fft = 16384 
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
    fft_freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

    W = np.zeros((len(c_freqs), n_mels))
    for i, f_c in enumerate(c_freqs):
        # Sample the Mel triangles at the octave center frequencies
        bin_idx = np.abs(fft_freqs - f_c).argmin()
        W[i, :] = mel_basis[:, bin_idx]

    # Energy conservation normalization (Unit Area)
    row_sums = W.sum(axis=1)
    W[row_sums > 0] = W[row_sums > 0] / row_sums[row_sums > 0][:, np.newaxis]

    return torch.from_numpy(W).float().to(device)

def convert_octave_to_msclap_mel(spectrogram_gpu, W_matrix):
    """
    Converts octave spectrogram to MS-CLAP Mel scale using a pre-computed 
    transition matrix.
    
    args:
     - spectrogram_gpu (torch.Tensor): [B, T, F_oct].
     - W_matrix (torch.Tensor): Pre-computed transition matrix [F_oct x 64].
        
    returns:
     - torch.Tensor: Formatted for HTS-AT [B, 1, T, 64].
    """
    # 1. Energy Projection via Matrix Multiplication
    # [B, T, F_octave] @ [F_octave, 64] -> [B, T, 64]
    x_mel = torch.matmul(spectrogram_gpu, W_matrix)

    # 2. Shape Formatting for HTS-AT encoder [B, C, T, F]
    x_mel = x_mel.unsqueeze(1) 

    # 3. Logarithmic Compression (matches standard CLAP/PANNs pipeline)
    x_log_mel = torch.log(torch.clamp(x_mel, min=1e-6))

    # 4. Instance-based Normalization
    # Rescales the input to zero mean / unit variance for the Transformer.
    mean = x_log_mel.mean(dim=(2, 3), keepdim=True)
    std = x_log_mel.std(dim=(2, 3), keepdim=True)
    x_norm = (x_log_mel - mean) / (std + 1e-6)

    return x_norm

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
    def __init__(self, classes, device='cpu', weights_path=None):
        super().__init__()
        self.classes = classes
        self.device = device
        self.classifier = torch.nn.Linear(1024, len(classes)).to(device)
        
        if weights_path and os.path.exists(weights_path):
            try:
                checkpoint = torch.load(weights_path, map_location=device)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    self.load_state_dict(checkpoint['state_dict'])
                else:
                    self.load_state_dict(checkpoint)
                print(f"✅ Models weights correctly loaded from: {weights_path}")
            except Exception as e:
                print(f"⚠️ Error during loading of weights: {e}")
        
        self.eval()
        
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.classifier.weight.dtype).to(self.classifier.weight.device)
        elif isinstance(x, torch.Tensor):
            x = x.to(self.device)

        # 🎯 FIX: Policy per forzare i NaN a 0.0 prima del calcolo
        if torch.isnan(x).any():
            # Opzione 1: Sostituzione con 0 (più sicura per la stabilità)
            x = torch.nan_to_num(x, nan=0.0)
            
            # Opzione 2 (Facoltativa): Logging per debugging
            # print(f"⚠️ Warning: NaNs detected in input batch and forced to 0.0")

        # Checks input dimension (Squeeze se necessario)
        if x.dim() > 2:
            x = x.squeeze(1)

        y = self.classifier(x)
        return y # torch.softmax(y, dim=0)

def train(tr_set, es_set, config, epochs, patience, device='cpu', classes=None, pretrained_path=None):
    """
    Funzione di training universale (CPU/GPU).
    """
    model = FinetunedModel(classes, device=device)
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict)
    optimizer, with_epochs = build_optimizer(config['optimizer'], model)
    
    # Se RR non usa epoche, forziamo a 1
    actual_epochs = epochs if with_epochs else 1
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    best_es_accuracy = 0.0
    best_params = model.state_dict()
    counter_es = 0

    for epoch in range(actual_epochs):
        model.train()
        for x, y in tr_set:
            x, y = x.to(device), y.to(device)
            if with_epochs:
                optimizer.zero_grad()
                h = model(x)
                loss = criterion(h, y)
                loss.backward()
                optimizer.step()
            else:
                optimizer(x, y) # Logica Ridge Regression
        
        if not with_epochs:
            optimizer.set_readout()
            
        model.eval()
        _, es_accuracy, _ = get_scores(model, es_set)
        
        if es_accuracy > best_es_accuracy:
            best_es_accuracy = es_accuracy
            best_params = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter_es = 0
        else:
            counter_es += 1
            if counter_es > patience:
                break
                
    model.load_state_dict(best_params)
    return model

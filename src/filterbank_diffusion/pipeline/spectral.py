import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import huggingface_hub
import transformers
import msclap

# Mantengo intatto il tuo blocco originale di iniezione dei percorsi
current_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if src_root not in sys.path: 
    sys.path.insert(0, src_root)

# 🎯 PRODUCTION GLOBAL VARIABLE
# If True, the audio encoder will be patched to accept pre-computed Mel spectrograms
INJECT_OCTAVE = os.environ.get("INJECT_OCTAVE", "False").lower() == "true"

# 🎯 MONKEY PATCH LOGIC
def universal_path_redirect(*args, **kwargs):
    """
    Redirects HuggingFace and MSCLAP download requests to local filesystem paths. 
    This prevents the library from attempting to connect to external hubs, 
    using environment variables to locate local weights and encoders.

    args:
     - *args: Variable length argument list from the original download function;
     - **kwargs: Arbitrary keyword arguments (e.g., 'filename').

    returns:
     - str: The local path to the model weights.
    """
    rank = os.environ.get('SLURM_PROCID', '0')
    weights_path = os.getenv("LOCAL_CLAP_WEIGHTS_PATH")
    text_path = os.getenv("CLAP_TEXT_ENCODER_PATH")

    # Redirect MSCLAP weights specifically
    if any(x for x in args if 'msclap' in str(x)) or 'CLAP_weights' in str(kwargs):
        return weights_path

    # Redirect general HuggingFace/Transformers files (like RoBERTa config/vocab)
    filename = kwargs.get('filename') or (args[1] if len(args) > 1 else None)
    
    if filename and text_path:
        forced_target = os.path.join(text_path, str(filename))
        # Log patch operations only if diagnostic VERBOSE is active
        if VERBOSE:
            print(f"🎯 [Rank {rank}] FIREWALL REDIRECT: {filename} -> {forced_target}", flush=True)
        return forced_target

    return text_path

# Applying path redirects
huggingface_hub.hf_hub_download = universal_path_redirect
transformers.utils.hub.cached_file = universal_path_redirect
transformers.utils.hub.hf_hub_download = universal_path_redirect
msclap.CLAPWrapper.hf_hub_download = universal_path_redirect

# ==============================================================================
# 💉 NEW PATCH: AUDIO ENCODER CLASS INJECTION (Targeting HTSAT_N_Level)
# ==============================================================================
if INJECT_OCTAVE:
    try:
        # Based on actual msclap source code hierarchy
        from msclap.models.htsat import HTSAT_N_Level
        
        def patched_forward(self, x):
            """
            Monkey patch for the HTSAT_N_Level forward method.
            If x is a 4D tensor [B, 1, T, 64], we bypass the 
            spectrogram_extractor (line 849) and call the backbone directly.
            """
            # Check if input is our pre-computed Mel spectrogram
            if isinstance(x, torch.Tensor) and x.ndim == 4:
                # We skip line 849 (spectrogram_extractor) and 
                # line 850 (logmel_extractor).
                # We jump directly to the transformer/convolutions layers.
                return self.forward_features(x)
            
            # Standard path for 1D audio waveforms
            return self.original_forward(x)

        # Apply the patch to the core engine class
        if not hasattr(HTSAT_N_Level, 'original_forward'):
            HTSAT_N_Level.original_forward = HTSAT_N_Level.forward
            HTSAT_N_Level.forward = patched_forward
            
        if VERBOSE:
            print("💉 MSCLAP PATCH: HTSAT_N_Level 'forward' successfully bypassed.")
            
    except ImportError:
        if VERBOSE:
            print("⚠️ WARNING: Could not find HTSAT_N_Level in msclap.models.htsat.")
# ==============================================================================


# Importiamo i moduli di elaborazione spettrale e l'inizializzatore con la patch dal tuo file core models
from src.models import spectrogram_n_octaveband_generator_gpu, convert_octave_to_msclap_mel, CLAP_initializer

class OnlineSpectrogramPipeline(nn.Module):
    """
    GPU-accelerated spectral transformation engine. Initializes CLAP cleanly
    via the native project initialization patch, using variables delegated to the .sh script.
    """
    def __init__(self, weights_path, sample_rate=51200, device='cuda'):
        super().__init__()
        self.sample_rate = sample_rate
        
        # Determiniamo il flag cuda coerentemente con il device del modello
        use_cuda = True if 'cuda' in str(device) else False
        
        # Inizializziamo l'istanza CLAP sfruttando la funzione patchata ufficiale del progetto.
        # Questa leggerà autonomamente le variabili d'ambiente esportate dallo script .sh
        clap_object, _, _ = CLAP_initializer(device=device, use_cuda=use_cuda)
        
        # Estraiamo il backbone HTS-AT già configurato, associato e protetto dalla classe madre
        self.htsat = clap_object.clap.audio_encoder.base.htsat
        self.htsat.to(device)
        self.htsat.eval()
        
        # Congeliamo i parametri in fase di addestramento/validazione della U-Net
        for param in self.htsat.parameters():
            param.requires_grad = False

    def forward(self, raw_audio_batch, format_id, fraction_id, device='cuda'):
        """
        args:
         - raw_audio_batch: 1D audio tensors batch from CPU DataLoader [B, 358400]
         - format_id: Target audio format codec (0=WAV, 1=MP3)
         - fraction_id: Active 1/n octave denominator (1, 3, 6, 12, 16, 24, 32)
         - device: Active execution device GPU
        returns:
         - x_native_norm: Pristine targets extracted using CLAP's original sub-layers [B, 1, 64, 700]
         - conditioning_C: Fused geometric spatial mask anchor [B, 1, 329, 700]
        """
        audio_signal = raw_audio_batch.to(torch.float32).to(device)
        audio_signal = torch.nan_to_num(audio_signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 1. Stochastic lossy codec simulation directly in RAM
        if format_id == 1: 
            fft_data = torch.fft.rfft(audio_signal, dim=-1)
            nyquist = self.sample_rate / 2.0
            cutoff_bin = int((16000.0 / nyquist) * fft_data.shape[-1])
            fft_data[:, cutoff_bin:] = 0.0 
            audio_signal = torch.fft.irfft(fft_data, dim=-1)
            audio_signal += torch.randn_like(audio_signal) * 1e-4

        # 2. Replicate CLAP Native Preprocessing pipeline for x_0 target
        with torch.no_grad():
            x_stft = self.htsat.spectrogram_extractor(audio_signal)
            x_native_logmel = self.htsat.logmel_extractor(x_stft)
            x_native_norm = self.htsat.bn0(x_native_logmel.transpose(1, 3)).transpose(1, 3)
        
        x_native_norm = x_native_norm.permute(0, 1, 3, 2) 

        # 3. Generate low-resolution structural condition filterbank C
        octave_spec = spectrogram_n_octaveband_generator_gpu(
            wav_batch=audio_signal,
            sampling_rate=self.sample_rate,
            n_octave=fraction_id,
            center_freqs=None,
            ref=2e-5,
            device=device
        )
        
        octave_spec = octave_spec.permute(0, 2, 1)
        conditioning_C = convert_octave_to_msclap_mel(octave_spec, target_mels=329)
        conditioning_C = conditioning_C.permute(0, 1, 3, 2) # Shape: [B, 1, 329, T_blocks]
        
        # Allineiamo la dimensione temporale preservando l'altezza nativa a 329 canali
        if conditioning_C.shape[-1] != x_native_norm.shape[-1]:
            conditioning_C = F.interpolate(
                conditioning_C,
                size=(329, x_native_norm.shape[-1]),
                mode='bilinear',
                align_corners=False
            )

        return x_native_norm, conditioning_C

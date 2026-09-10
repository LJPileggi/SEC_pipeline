#!/bin/bash
#SBATCH --job-name=inspect_clap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_BrISkite_0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_inspect_$SLURM_JOB_ID"
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
CLAP_TEXT_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/text_encoder"

mkdir -p "$TEMP_DIR/weights"
mkdir -p "$TEMP_DIR/numba_cache"

cp "$CLAP_SCRATCH_WEIGHTS" "$TEMP_DIR/weights/CLAP_weights_2023.pth" 2>/dev/null

cat << 'EOF' > "$TEMP_DIR/run_inspect.py"
import os
import sys

os.environ["NUMBA_CACHE_DIR"] = "/tmp_data/numba_cache"
os.environ["MPLCONFIGDIR"] = "/tmp_data/numba_cache"

sys.path.insert(0, "/app")

import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

import huggingface_hub
import transformers
import msclap

# 1. Bypass offline del caricamento pesi HuggingFace
def universal_path_redirect(*args, **kwargs):
    weights_path = os.getenv("LOCAL_CLAP_WEIGHTS_PATH")
    text_path = os.getenv("CLAP_TEXT_ENCODER_PATH")
    if any(x for x in args if 'msclap' in str(x)) or 'CLAP_weights' in str(kwargs):
        return weights_path
    filename = kwargs.get('filename') or (args[1] if len(args) > 1 else None)
    if filename and text_path and os.path.exists(os.path.join(text_path, str(filename))):
        return os.path.join(text_path, str(filename))
    return text_path

huggingface_hub.hf_hub_download = universal_path_redirect
transformers.utils.hub.cached_file = universal_path_redirect
transformers.utils.hub.hf_hub_download = universal_path_redirect
msclap.CLAPWrapper.hf_hub_download = universal_path_redirect

# Se il text encoder non ha i pesi scaricati offline, facciamo un dummy mock per consentire a CLAP(...) di istanziarsi
try:
    orig_from_pretrained = transformers.AutoModel.from_pretrained
    def safe_from_pretrained(*args, **kwargs):
        try:
            return orig_from_pretrained(*args, **kwargs)
        except Exception:
            class DummyTextModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.config = type('cfg', (), {'hidden_size': 768})()
                def forward(self, *a, **k):
                    return type('out', (), {'last_hidden_state': torch.zeros(1, 10, 768)})()
            return DummyTextModel()
    transformers.AutoModel.from_pretrained = safe_from_pretrained
except Exception:
    pass

from msclap import CLAP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

# Inizializzazione CLAP standard pura nativa
clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())
clap_model.clap.to(device)
htsat = clap_model.clap.audio_encoder.base.htsat

print("\n" + "="*70)
print("1. SORGENTE NATIVO DI HTSAT.forward")
print("="*70)
try:
    print(inspect.getsource(htsat.forward))
except Exception as e:
    print(f"Errore ispezione htsat.forward: {e}")

print("\n" + "="*70)
print("2. SORGENTE NATIVO DI AudioEncoder.base.forward")
print("="*70)
try:
    print(inspect.getsource(clap_model.clap.audio_encoder.base.forward))
except Exception as e:
    print(f"Errore ispezione base.forward: {e}")

print("\n" + "="*70)
print("3. TRACCIAMENTO SULLE FORME DEI TENSORI (SR = 52.100 Hz)")
print("="*70)

shapes_log = []

def make_hook(name):
    def hook(module, inp, out):
        in_shape = inp[0].shape if isinstance(inp, tuple) and len(inp) > 0 and hasattr(inp[0], 'shape') else 'N/A'
        out_shape = out.shape if hasattr(out, 'shape') else (out[0].shape if isinstance(out, tuple) else 'tuple/dict')
        shapes_log.append(f"   ↳ [{name}] In: {in_shape} -> Out: {out_shape}")
    return hook

# Hook per i moduli PyTorch
htsat.spectrogram_extractor.register_forward_hook(make_hook("spectrogram_extractor"))
htsat.logmel_extractor.register_forward_hook(make_hook("logmel_extractor"))
htsat.bn0.register_forward_hook(make_hook("bn0"))

# Wrapping manuale per reshape_wav2img (è un metodo python nativo, non un nn.Module)
orig_reshape = htsat.reshape_wav2img
def wrapped_reshape(x):
    in_s = x.shape
    out = orig_reshape(x)
    shapes_log.append(f"   ↳ [reshape_wav2img] In: {in_s} -> Out: {out.shape}")
    return out
htsat.reshape_wav2img = wrapped_reshape

sr = 52100
durations = [1.0, 3.0, 5.0, 7.0, 10.0, 30.0]

for d in durations:
    shapes_log.clear()
    n_samples = int(d * sr)
    waveform = torch.randn(1, n_samples, device=device)
    print(f"\n▶ Test Durata {d:4.1f}s | Campioni: {n_samples} | Formula teorica T: {(n_samples // 320) + 1}")
    try:
        with torch.no_grad():
            out = clap_model.clap.audio_encoder(waveform)
        emb = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(emb, dict):
            emb = emb.get('embedding', emb.get('clipwise_output'))
        if emb.ndim > 2:
            emb = emb.squeeze(1)
        print(f"   ✅ Forward riuscito! Embedding: {emb.shape}")
        for line in shapes_log:
            print(line)
    except Exception as e:
        print(f"   ❌ ERRORE: {e}")
        for line in shapes_log:
            print(line)

print("\n" + "="*70)
EOF

export LOCAL_CLAP_WEIGHTS_PATH="$TEMP_DIR/weights/CLAP_weights_2023.pth"
export CLAP_TEXT_ENCODER_PATH="$CLAP_TEXT_PATH"
export NUMBA_CACHE_DIR="$TEMP_DIR/numba_cache"

singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEMP_DIR:/tmp_data" \
    --bind "$(pwd):/app" --pwd "/app" \
    "$SIF_FILE" \
    python3 /tmp_data/run_inspect.py

rm -rf "$TEMP_DIR"

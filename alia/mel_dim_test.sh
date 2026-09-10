#!/bin/bash
#SBATCH --job-name=inspect_clap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:08:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_BrISkite_0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_inspect_$SLURM_JOB_ID"
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
ROBERTA_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/roberta-base"

mkdir -p "$TEMP_DIR/weights"
mkdir -p "$TEMP_DIR/roberta-base"
mkdir -p "$TEMP_DIR/numba_cache"

echo "📦 Stage-in: Pesi CLAP e RoBERTa..."
cp "$CLAP_SCRATCH_WEIGHTS" "$TEMP_DIR/weights/CLAP_weights_2023.pth" 2>/dev/null
cp -r "$ROBERTA_PATH/." "$TEMP_DIR/roberta-base/" 2>/dev/null

cat << 'EOF' > "$TEMP_DIR/run_inspect.py"
import os
import sys

os.environ["NUMBA_CACHE_DIR"] = "/tmp_data/numba_cache"
os.environ["MPLCONFIGDIR"] = "/tmp_data/numba_cache"

sys.path.insert(0, "/app")

import inspect
import torch
import torch.nn.functional as F

import huggingface_hub
import transformers
import msclap

def universal_path_redirect(*args, **kwargs):
    weights_path = os.getenv("LOCAL_CLAP_WEIGHTS_PATH")
    text_path = os.getenv("CLAP_TEXT_ENCODER_PATH")

    if any(x for x in args if 'msclap' in str(x)) or 'CLAP_weights' in str(kwargs):
        return weights_path

    filename = kwargs.get('filename') or (args[1] if len(args) > 1 else None)
    if filename and text_path:
        target = os.path.join(text_path, str(filename))
        if os.path.exists(target):
            return target

    return text_path

huggingface_hub.hf_hub_download = universal_path_redirect
transformers.utils.hub.cached_file = universal_path_redirect
transformers.utils.hub.hf_hub_download = universal_path_redirect
msclap.CLAPWrapper.hf_hub_download = universal_path_redirect

from src.models import CLAP_initializer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🔧 Device selezionato: {device}")

# Inizializzazione ufficiale della pipeline del progetto (con RoBERTa)
clap_model, _, _ = CLAP_initializer(device=device, use_cuda=True)
clap_model.clap.to(device)
htsat = clap_model.clap.audio_encoder.base.htsat
htsat.eval()

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

htsat.spectrogram_extractor.register_forward_hook(make_hook("spectrogram_extractor"))
htsat.logmel_extractor.register_forward_hook(make_hook("logmel_extractor"))
htsat.bn0.register_forward_hook(make_hook("bn0"))

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

export HF_HUB_OFFLINE=1
export CLAP_TEXT_ENCODER_PATH="/tmp_data/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/weights/CLAP_weights_2023.pth"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"

echo "🚀 Esecuzione test dimensioni..."
singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEMP_DIR:/tmp_data" \
    --bind "$(pwd):/app" --pwd "/app" \
    "$SIF_FILE" \
    python3 /tmp_data/run_inspect.py

rm -rf "$TEMP_DIR"

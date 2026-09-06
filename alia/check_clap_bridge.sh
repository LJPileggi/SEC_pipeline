#!/bin/bash
#SBATCH --job-name=check_bridge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_BrISkite_0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_diag_$SLURM_JOB_ID"
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
CLAP_BN0_CONSTANTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/clap_bn0_constants.npz"
CLAP_TEXT_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/text_encoder"
DATASEC_GLOBAL="/leonardo_scratch/large/userexternal/$USER/dataSEC"
MODELS_GLOBAL="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.models/diff_model"

mkdir -p "$TEMP_DIR/work_dir/weights"
mkdir -p "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav"
mkdir -p "$TEMP_DIR/models/diff_model"
mkdir -p "$TEMP_DIR/numba_cache"

echo "📦 Stage-in: Setup diagnostico..."
cp "$CLAP_SCRATCH_WEIGHTS" "$TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth" 2>/dev/null
[ -f "$CLAP_BN0_CONSTANTS" ] && cp "$CLAP_BN0_CONSTANTS" "$TEMP_DIR/work_dir/weights/clap_bn0_constants.npz" 2>/dev/null
cp "$DATASEC_GLOBAL/RAW_DATASET/raw_wav"/*.h5 "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav/" 2>/dev/null

if [ -f "$MODELS_GLOBAL/unet_epoch_89.pt" ]; then
    cp "$MODELS_GLOBAL/unet_epoch_89.pt" "$TEMP_DIR/models/diff_model/unet_epoch_89.pt"
elif [ -d "$MODELS_GLOBAL" ]; then
    cp "$MODELS_GLOBAL"/*.pt "$TEMP_DIR/models/diff_model/" 2>/dev/null
fi

cat << 'EOF' > "$TEMP_DIR/diagnose_clap_bridge.py"
import os
import sys

sys.path.insert(0, "/app")

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
        return os.path.join(text_path, str(filename))
    return text_path

huggingface_hub.hf_hub_download = universal_path_redirect
transformers.utils.hub.cached_file = universal_path_redirect
transformers.utils.hub.hf_hub_download = universal_path_redirect
msclap.CLAPWrapper.hf_hub_download = universal_path_redirect

import torch
import torch.nn.functional as F
import numpy as np
import h5py

from src.models import CLAP_initializer, convert_octave_to_msclap_mel, spectrogram_n_octaveband_generator_gpu
from src.filterbank_diffusion.models.unet import SpectrogramUNet
from src.filterbank_diffusion.models.diffusion import ConditionalGaussianDiffusion
from src.filterbank_diffusion.data.dataset import DistributedAudioRAWDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Device selezionato: {device}")

clap_model, _, _ = CLAP_initializer(device=device, use_cuda=True)
htsat = clap_model.clap.audio_encoder.base.htsat
htsat.eval()

# Frequenza nativa del dataset
sr_dataset = 52100

# Caricamento audio di test
raw_wav_dir = "/tmp_data/dataSEC/RAW_DATASET/raw_wav"
try:
    dataset = DistributedAudioRAWDataset(base_dir=raw_wav_dir, split="test", target_samples_per_class=5)
    raw_audio, _ = dataset[0]
    dataset.close()
except Exception as e:
    available_h5 = [f for f in os.listdir(raw_wav_dir) if f.endswith(".h5")]
    sample_h5_path = os.path.join(raw_wav_dir, available_h5[0])
    with h5py.File(sample_h5_path, "r") as hf:
        raw_audio = hf[list(hf.keys())[0]][:]

audio_tensor = torch.as_tensor(raw_audio, dtype=torch.float32).flatten()

# Normalizzazione temporale a 7 secondi nativi a 52100 Hz
target_samples = int(sr_dataset * 7.0)
if audio_tensor.numel() < target_samples:
    audio_tensor = F.pad(audio_tensor, (0, target_samples - audio_tensor.numel()))
else:
    audio_tensor = audio_tensor[:target_samples]

audio_tensor = audio_tensor.unsqueeze(0).to(device)

print("\n" + "="*65)
print("🔍 TEST 1: ESTRAZIONE NATIVA UFFICIALE")
print("="*65)

with torch.no_grad():
    out_native = clap_model.clap.audio_encoder(audio_tensor)
    vec_official = out_native[0] if isinstance(out_native, (tuple, list)) else out_native
    if isinstance(vec_official, dict):
        vec_official = vec_official.get('embedding', vec_official.get('clipwise_output'))
    if vec_official.ndim > 2:
        vec_official = vec_official.squeeze(1)
    emb_official = F.normalize(vec_official, p=2, dim=-1)

    # Estrazione Log-Mel nativo dai layer STFT -> LogMel -> bn0
    x_stft = htsat.spectrogram_extractor(audio_tensor)
    x_logmel = htsat.logmel_extractor(x_stft)
    x_norm = htsat.bn0(x_logmel.transpose(1, 3)).transpose(1, 3)
    
    print(f"✅ Embedding nativo estratto. Shape: {emb_official.shape}")
    print(f"📐 x_norm post-bn0 shape: {x_norm.shape} (T={x_norm.shape[2]}, F={x_norm.shape[3]})")

print("\n" + "="*65)
print("🔍 TEST 2: ALLINEAMENTO IDENTITÀ (Crop T=1024 e Crop T=700)")
print("="*65)

with torch.no_grad():
    # Test 2A: Finestra a 1024 frame
    x_norm_1024 = x_norm[:, :, :1024, :]
    x_ready_1024 = htsat.reshape_wav2img(x_norm_1024)
    out_1024 = clap_model.clap.audio_encoder(x_ready_1024)
    vec_1024 = out_1024[0] if isinstance(out_1024, (tuple, list)) else out_1024
    if isinstance(vec_1024, dict):
        vec_1024 = vec_1024.get('embedding', vec_1024.get('clipwise_output'))
    if vec_1024.ndim > 2:
        vec_1024 = vec_1024.squeeze(1)
    emb_1024 = F.normalize(vec_1024, p=2, dim=-1)

    sim_1024 = F.cosine_similarity(emb_official, emb_1024, dim=-1).item()
    print(f"🎯 Coseno Nativo Full vs Iniezione Log-Mel Nativo (1024 frame): {sim_1024:.6f}")

    # Test 2B: Finestra a 700 frame (la stessa che riceve la U-Net)
    x_norm_700 = x_norm[:, :, :700, :]
    x_ready_700 = htsat.reshape_wav2img(x_norm_700)
    out_700 = clap_model.clap.audio_encoder(x_ready_700)
    vec_700 = out_700[0] if isinstance(out_700, (tuple, list)) else out_700
    if isinstance(vec_700, dict):
        vec_700 = vec_700.get('embedding', vec_700.get('clipwise_output'))
    if vec_700.ndim > 2:
        vec_700 = vec_700.squeeze(1)
    emb_ref_700 = F.normalize(vec_700, p=2, dim=-1)

    sim_700_vs_full = F.cosine_similarity(emb_official, emb_ref_700, dim=-1).item()
    print(f"🎯 Coseno Nativo Full vs Nativo Troncato a 700 frame:         {sim_700_vs_full:.6f}")

print("\n" + "="*65)
print("🔍 TEST 3: VERIFICA SPETTROGRAMMA RICOSTRUITO (DDIM EPOCA 89)")
print("="*65)

ckpt_dir = "/tmp_data/models/diff_model"
pts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
pts_sorted = sorted(pts, key=lambda x: int(x.replace("unet_epoch_", "").replace(".pt", "")))
target_ckpt = os.path.join(ckpt_dir, pts_sorted[-1])
print(f"📦 Checkpoint caricato: {target_ckpt}")

unet = SpectrogramUNet(base_channels=64, emb_dim=256).to(device)
ckpt = torch.load(target_ckpt, map_location=device)
unet.load_state_dict(ckpt['model_state_dict'])
diffusion = ConditionalGaussianDiffusion(unet_model=unet, timesteps=1000).to(device)

# Target reale nello spazio U-Net a 700 frame: [1, 1, 64, 700]
x_target_unet = x_norm_700.permute(0, 1, 3, 2)

with torch.no_grad():
    for frac in [3, 32]:
        # Calcolo rigoroso a frequenza nativa di 52100 Hz
        spec_octave = spectrogram_n_octaveband_generator_gpu(
            audio_tensor, sampling_rate=sr_dataset, n_octave=frac, center_freqs=None, ref=2e-5, device=device
        )
        spec_octave = spec_octave.permute(0, 2, 1)

        # Resampling 2D normalizzato bn0 (target: [1, 1, 64, 700])
        x_cond = convert_octave_to_msclap_mel(spec_octave, target_mels=64, target_time=700)
        frac_t = torch.tensor([float(frac)], device=device)

        # Campionamento DDIM (output: [1, 1, 64, 700])
        mel_rec = diffusion.sample_ddim(x_cond, fraction_id=frac_t, ddim_steps=25)

        # Distanza di Frobenius rispetto al target 700
        frob = torch.norm(x_target_unet - mel_rec, p='fro').item()

        # TRASPOSIZIONE CORRETTA PER HTS-AT: da [1, 1, 64, 700] a [1, 1, 700, 64]
        mel_rec_htsat = mel_rec.permute(0, 1, 3, 2)
        x_rec_ready = htsat.reshape_wav2img(mel_rec_htsat)

        out_rec = clap_model.clap.audio_encoder(x_rec_ready)
        vec_rec = out_rec[0] if isinstance(out_rec, (tuple, list)) else out_rec
        if isinstance(vec_rec, dict):
            vec_rec = vec_rec.get('embedding', vec_rec.get('clipwise_output'))
        if vec_rec.ndim > 2:
            vec_rec = vec_rec.squeeze(1)
        emb_rec = F.normalize(vec_rec, p=2, dim=-1)

        sim_vs_700 = F.cosine_similarity(emb_ref_700, emb_rec, dim=-1).item()
        sim_vs_full = F.cosine_similarity(emb_official, emb_rec, dim=-1).item()

        print(f"🎯 Frazione 1/{frac:02d} | Frobenius: {frob:6.2f} | Coseno vs Nativo 700: {sim_vs_700:.6f} | Coseno vs Full: {sim_vs_full:.6f}")

print("\n" + "="*65)
print("🏁 DIAGNOSTICA COMPLETATA")
print("="*65)
EOF

export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
export LOCAL_CLAP_BN0_CONSTANTS_PATH="/tmp_data/work_dir/weights/clap_bn0_constants.npz"
export CLAP_TEXT_ENCODER_PATH="$CLAP_TEXT_PATH"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export INJECT_OCTAVE="True"
export VERBOSE="False"

echo "🚀 Esecuzione script diagnostico all'interno del container..."
singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEMP_DIR:/tmp_data" \
    --bind "$(pwd):/app" --pwd "/app" \
    "$SIF_FILE" \
    python3 /tmp_data/diagnose_clap_bridge.py

rm -rf "$TEMP_DIR"

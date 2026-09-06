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

echo "📦 Stage-in: Setup..."
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

# Normalizzazione all'esatta finestra di CLAP (1024 frame * 320 hop = 327.680 campioni)
target_len = 1024 * 320
if audio_tensor.numel() < target_len:
    audio_tensor = F.pad(audio_tensor, (0, target_len - audio_tensor.numel()))
else:
    audio_tensor = audio_tensor[:target_len]

audio_tensor = audio_tensor.unsqueeze(0).to(device)

print("\n" + "="*65)
print("🔍 TEST 1: ESTRAZIONE NATIVA UFFICIALE")
print("="*65)

with torch.no_grad():
    # 1. Forward nativo su forma d'onda grezza
    out_native = clap_model.clap.audio_encoder(audio_tensor)
    vec_official = out_native[0] if isinstance(out_native, (tuple, list)) else out_native
    if isinstance(vec_official, dict):
        vec_official = vec_official.get('embedding', vec_official.get('clipwise_output'))
    if vec_official.ndim > 2:
        vec_official = vec_official.squeeze(1)
    emb_official = F.normalize(vec_official, p=2, dim=-1)

    # 2. Estrazione esatta dei blocchi interni HTS-AT
    x_stft = htsat.spectrogram_extractor(audio_tensor)
    x_logmel = htsat.logmel_extractor(x_stft)
    x_norm = htsat.bn0(x_logmel.transpose(1, 3)).transpose(1, 3)  # [1, 1, 1024, 64]
    
    print(f"✅ Embedding nativo estratto. Shape: {emb_official.shape}")
    print(f"📐 Dimensione x_norm interno: {x_norm.shape} (T={x_norm.shape[2]}, F={x_norm.shape[3]})")

print("\n" + "="*65)
print("🔍 TEST 2: ALLINEAMENTO IDENTITÀ (Mel Nativo Iniettato)")
print("="*65)

with torch.no_grad():
    # x_norm è [1, 1, 1024, 64]. reshape_wav2img riceve [B, 1, T, F]
    x_ready_native = htsat.reshape_wav2img(x_norm)
    out_injected = clap_model.clap.audio_encoder(x_ready_native)
    vec_injected = out_injected[0] if isinstance(out_injected, (tuple, list)) else out_injected
    if isinstance(vec_injected, dict):
        vec_injected = vec_injected.get('embedding', vec_injected.get('clipwise_output'))
    if vec_injected.ndim > 2:
        vec_injected = vec_injected.squeeze(1)
    emb_injected = F.normalize(vec_injected, p=2, dim=-1)

    identity_sim = F.cosine_similarity(emb_official, emb_injected, dim=-1).item()
    print(f"🎯 Similarità Coseno (Nativo Ufficiale vs Log-Mel Nativo Iniettato): {identity_sim:.6f}")

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

# Creazione target pulito a 700 frame (spazio U-Net)
x_0_pristine = x_norm[:, :, :700, :].permute(0, 1, 3, 2)  # [1, 1, 64, 700]

with torch.no_grad():
    for frac in [3, 32]:
        spec_octave = spectrogram_n_octaveband_generator_gpu(
            audio_tensor, sampling_rate=48000, n_octave=frac, center_freqs=None, ref=2e-5, device=device
        )
        spec_octave = spec_octave.permute(0, 2, 1)

        x_cond = convert_octave_to_msclap_mel(spec_octave, target_mels=64, target_time=700)
        frac_t = torch.tensor([float(frac)], device=device)

        # mel_rec esce dalla U-Net come [1, 1, 64, 700] (F=64, T=700)
        mel_rec = diffusion.sample_ddim(x_cond, fraction_id=frac_t, ddim_steps=25)
        frob = torch.norm(x_0_pristine - mel_rec, p='fro').item()

        # TRASPOSIZIONE RIGOROSA PER HTS-AT: da [1, 1, 64, 700] a [1, 1, 700, 64]
        mel_rec_htsat = mel_rec.permute(0, 1, 3, 2)

        # reshape_wav2img riceve [1, 1, 700, 64], rispetta T=700 <= 1024 e fa zero-pad a 1024 internamente
        x_rec_ready = htsat.reshape_wav2img(mel_rec_htsat)

        out_rec = clap_model.clap.audio_encoder(x_rec_ready)
        vec_rec = out_rec[0] if isinstance(out_rec, (tuple, list)) else out_rec
        if isinstance(vec_rec, dict):
            vec_rec = vec_rec.get('embedding', vec_rec.get('clipwise_output'))
        if vec_rec.ndim > 2:
            vec_rec = vec_rec.squeeze(1)
        emb_rec = F.normalize(vec_rec, p=2, dim=-1)

        sim_rec = F.cosine_similarity(emb_official, emb_rec, dim=-1).item()
        print(f"🎯 Frazione 1/{frac:02d} | Frobenius: {frob:6.2f} | Coseno vs Nativo: {sim_rec:.6f}")

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

echo "🚀 Esecuzione script diagnostico..."
singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEMP_DIR:/tmp_data" \
    --bind "$(pwd):/app" --pwd "/app" \
    "$SIF_FILE" \
    python3 /tmp_data/diagnose_clap_bridge.py

rm -rf "$TEMP_DIR"

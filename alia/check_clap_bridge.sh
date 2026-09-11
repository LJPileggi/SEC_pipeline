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
    cp "$MODELS_GLOBAL/unet_epoch_89.pt" "$TEMP_DIR/models/diff_model/unet_epoch_124.pt"
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

from src.models import CLAP_initializer, convert_octave_to_msclap_mel, \
    spectrogram_n_octaveband_generator_gpu, extract_clap_embedding_from_reconstructed_mel
from src.filterbank_diffusion.models.unet import SpectrogramUNet
from src.filterbank_diffusion.models.diffusion import ConditionalGaussianDiffusion
from src.filterbank_diffusion.data.dataset import DistributedAudioRAWDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Device selezionato: {device}")

clap_model, _, _ = CLAP_initializer(device=device, use_cuda=True)
htsat = clap_model.clap.audio_encoder.base.htsat
htsat.eval()

# Sampling rate nativo del registratore
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

# Normalizzazione temporale rigorosa a 7.0 secondi nativi a 52.100 Hz (364.700 campioni)
target_samples = int(sr_dataset * 7.0)
if audio_tensor.numel() < target_samples:
    audio_tensor = F.pad(audio_tensor, (0, target_samples - audio_tensor.numel()))
else:
    audio_tensor = audio_tensor[:target_samples]

audio_tensor = audio_tensor.unsqueeze(0).to(device)

print("\n" + "="*65)
print("🔍 TEST 1: ESTRAZIONE NATIVA UFFICIALE (Audio Grezzo 7s)")
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
    print(f"📐 x_norm post-bn0 nativo shape: {x_norm.shape} (T={x_norm.shape[2]}, F={x_norm.shape[3]})")

print("\n" + "="*65)
print("🔍 TEST 2: ALLINEAMENTO IDENTITÀ (extract_clap_embedding vs Ufficiale)")
print("="*65)

with torch.no_grad():
    # Convertiamo x_norm post-bn0 nel formato di input U-Net: [1, 1, 64, 1140]
    mel_native_unet_space = x_norm.permute(0, 1, 3, 2).contiguous()

    # Passaggio attraverso extract_clap_embedding_from_reconstructed_mel
    emb_bridge = extract_clap_embedding_from_reconstructed_mel(
        mel_reconstructed=mel_native_unet_space,
        clap_model=clap_model,
        target_time=1140,
        device=device
    )

    sim_identity = F.cosine_similarity(emb_official, emb_bridge, dim=-1).item()
    print(f"🎯 Coseno Nativo Ufficiale vs Bridge extract_clap_embedding: {sim_identity:.6f}")
    if abs(sim_identity - 1.0) < 1e-4:
        print("   🏆 VERIFICA SUPERATA: Il bridging matematico di HTS-AT è identico all'ufficiale!")
    else:
        print(f"   ⚠️ ATTENZIONE: Delta di allineamento = {abs(sim_identity - 1.0):.6f}")

print("\n" + "="*65)
print("🔍 TEST 3: VERIFICA SPETTROGRAMMA RICOSTRUITO DA U-NET")
print("="*65)

ckpt_dir = "/tmp_data/models/diff_model"
pts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
if not pts:
    print(f"⚠️ Nessun checkpoint trovato in {ckpt_dir}. Salto il Test 3.")
else:
    pts_sorted = sorted(pts, key=lambda x: int(x.replace("unet_epoch_", "").replace(".pt", "")))
    target_ckpt = os.path.join(ckpt_dir, pts_sorted[-1])
    print(f"📦 Checkpoint caricato: {target_ckpt}")

    unet = SpectrogramUNet(base_channels=64, emb_dim=256).to(device)
    ckpt = torch.load(target_ckpt, map_location=device)
    unet.load_state_dict(ckpt['model_state_dict'])
    diffusion = ConditionalGaussianDiffusion(unet_model=unet, timesteps=1000).to(device)

    # Riferimento per il calcolo della distanza di Frobenius
    target_time_eval = 1152 if mel_native_unet_space.shape[-1] >= 1152 else mel_native_unet_space.shape[-1]
    x_target_ref = mel_native_unet_space[:, :, :, :target_time_eval]

    with torch.no_grad():
        for frac in [3, 32]:
            spec_octave = spectrogram_n_octaveband_generator_gpu(
                audio_tensor, sampling_rate=sr_dataset, n_octave=frac, center_freqs=None, ref=2e-5, device=device
            )
            spec_octave = spec_octave.permute(0, 2, 1)

            # Resampling 2D normalizzato bn0 sulla nuova griglia da 1152 frame
            x_cond = convert_octave_to_msclap_mel(spec_octave, target_mels=64, target_time=1152)
            frac_t = torch.tensor([float(frac)], device=device)

            # Campionamento DDIM
            mel_rec = diffusion.sample_ddim(x_cond, fraction_id=frac_t, ddim_steps=25)

            # Distanza di Frobenius sui frame sovrapposti
            frob = torch.norm(x_target_ref - mel_rec[:, :, :, :target_time_eval], p='fro').item()

            # Estrazione embedding con la funzione di produzione
            emb_rec = extract_clap_embedding_from_reconstructed_mel(
                mel_reconstructed=mel_rec,
                clap_model=clap_model,
                target_time=1140,
                device=device
            )

            sim_rec_vs_full = F.cosine_similarity(emb_official, emb_rec, dim=-1).item()
            print(f"🎯 Frazione 1/{frac:02d} | Frobenius: {frob:6.2f} | Coseno Rec vs Full Nativo: {sim_rec_vs_full:.6f}")

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

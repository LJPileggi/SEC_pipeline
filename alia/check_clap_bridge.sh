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

echo "📦 Stage-in: Configurazione ambiente di diagnostica..."
cp "$CLAP_SCRATCH_WEIGHTS" "$TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth" 2>/dev/null
[ -f "$CLAP_BN0_CONSTANTS" ] && cp "$CLAP_BN0_CONSTANTS" "$TEMP_DIR/work_dir/weights/clap_bn0_constants.npz" 2>/dev/null

# Copia dei file audio HDF5
cp "$DATASEC_GLOBAL/RAW_DATASET/raw_wav"/*.h5 "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav/" 2>/dev/null

# Copia del checkpoint epoca 89
if [ -f "$MODELS_GLOBAL/unet_epoch_89.pt" ]; then
    cp "$MODELS_GLOBAL/unet_epoch_89.pt" "$TEMP_DIR/models/diff_model/unet_epoch_89.pt"
elif [ -d "$MODELS_GLOBAL" ]; then
    cp "$MODELS_GLOBAL"/*.pt "$TEMP_DIR/models/diff_model/" 2>/dev/null
fi

# Scrittura dello script Python temporaneo
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

raw_wav_dir = "/tmp_data/dataSEC/RAW_DATASET/raw_wav"
try:
    dataset = DistributedAudioRAWDataset(base_dir=raw_wav_dir, split="test", target_samples_per_class=5)
    raw_audio, _ = dataset[0]
    dataset.close()
    print("📖 Audio di test caricato tramite DistributedAudioRAWDataset.")
except Exception as e:
    print(f"⚠️ DistributedAudioRAWDataset non utilizzabile ({e}), lettura diretta via h5py...")
    available_h5 = [f for f in os.listdir(raw_wav_dir) if f.endswith(".h5")]
    if not available_h5:
        raise FileNotFoundError(f"Nessun file HDF5 trovato in {raw_wav_dir}")
    sample_h5_path = os.path.join(raw_wav_dir, available_h5[0])
    with h5py.File(sample_h5_path, "r") as hf:
        first_key = list(hf.keys())[0]
        raw_audio = hf[first_key][:]
    print(f"📖 Audio di test caricato da: {sample_h5_path}")

# Conversione a tensore
audio_tensor = torch.as_tensor(raw_audio, dtype=torch.float32).flatten()

# Normalizzazione a 7 secondi (48 kHz)
sr = 48000
target_len = sr * 7
if audio_tensor.numel() < target_len:
    audio_tensor = F.pad(audio_tensor, (0, target_len - audio_tensor.numel()))
else:
    audio_tensor = audio_tensor[:target_len]

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

    print(f"✅ Embedding nativo estratto. Shape: {emb_official.shape} | L2-norm: {torch.norm(emb_official).item():.4f}")

    x_stft = htsat.spectrogram_extractor(audio_tensor)
    x_logmel = htsat.logmel_extractor(x_stft)
    x_norm = htsat.bn0(x_logmel.transpose(1, 3)).transpose(1, 3) # Shape: [1, 1, 700, 64]
    x_0_pristine = x_norm.permute(0, 1, 3, 2)                    # Shape: [1, 1, 64, 700]

print("\n" + "="*65)
print("🔍 TEST 2: FORWARD PATCH DI CLAP SUL MEL NATIVO")
print("="*65)

with torch.no_grad():
    # x_0_pristine ha shape [1, 1, 64, 700] (F=64, T=700)
    mel_input = x_0_pristine.to(device)

    # Variante A: Con reshape_wav2img su [1, 1, 64, 700]
    x_ready_manual = htsat.reshape_wav2img(mel_input)
    out_manual = clap_model.clap.audio_encoder(x_ready_manual)
    vec_manual = out_manual[0] if isinstance(out_manual, (tuple, list)) else out_manual
    if isinstance(vec_manual, dict):
        vec_manual = vec_manual.get('embedding', vec_manual.get('clipwise_output'))
    if vec_manual.ndim > 2:
        vec_manual = vec_manual.squeeze(1)
    emb_manual = F.normalize(vec_manual, p=2, dim=-1)

    sim_manual = F.cosine_similarity(emb_official, emb_manual, dim=-1).item()
    print(f"• Similarità Coseno (Ufficiale vs Mel Nativo con reshape manuale): {sim_manual:.6f}")

    # Variante B: Passando direttamente mel_input senza reshape preliminare
    try:
        out_direct = clap_model.clap.audio_encoder(mel_input)
        vec_direct = out_direct[0] if isinstance(out_direct, (tuple, list)) else out_direct
        if isinstance(vec_direct, dict):
            vec_direct = vec_direct.get('embedding', vec_direct.get('clipwise_output'))
        if vec_direct.ndim > 2:
            vec_direct = vec_direct.squeeze(1)
        emb_direct = F.normalize(vec_direct, p=2, dim=-1)
        sim_direct = F.cosine_similarity(emb_official, emb_direct, dim=-1).item()
        print(f"• Similarità Coseno (Ufficiale vs Mel Nativo diretto):              {sim_direct:.6f}")
    except Exception as e:
        print(f"• Variante B non applicabile: {e}")

print("\n" + "="*65)
print("🔍 TEST 3: VERIFICA SULLO SPETTROGRAMMA RICOSTRUITO (DDIM EPOCA 89)")
print("="*65)

ckpt_dir = "/tmp_data/models/diff_model"
pts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
if not pts:
    print("❌ Nessun file checkpoint (.pt) trovato in /tmp_data/models/diff_model.")
    sys.exit(0)

pts_sorted = sorted(pts, key=lambda x: int(x.replace("unet_epoch_", "").replace(".pt", "")))
target_ckpt = os.path.join(ckpt_dir, pts_sorted[-1])
print(f"📦 Caricamento diffusion checkpoint: {target_ckpt}")

unet = SpectrogramUNet(base_channels=64, emb_dim=256).to(device)
ckpt = torch.load(target_ckpt, map_location=device)
unet.load_state_dict(ckpt['model_state_dict'])
diffusion = ConditionalGaussianDiffusion(unet_model=unet, timesteps=1000).to(device)

with torch.no_grad():
    for frac in [3, 32]:
        spec_octave = spectrogram_n_octaveband_generator_gpu(
            audio_tensor, sampling_rate=sr, n_octave=frac, center_freqs=None, ref=2e-5, device=device
        )
        spec_octave = spec_octave.permute(0, 2, 1)

        x_cond = convert_octave_to_msclap_mel(spec_octave, target_mels=64, target_time=700)
        frac_t = torch.tensor([float(frac)], device=device)

        # mel_rec ha shape [1, 1, 64, 700]
        mel_rec = diffusion.sample_ddim(x_cond, fraction_id=frac_t, ddim_steps=25)
        frob = torch.norm(x_0_pristine - mel_rec, p='fro').item()

        # Iniezione diretta senza permute invertito
        x_rec_ready = htsat.reshape_wav2img(mel_rec)
        out_rec = clap_model.clap.audio_encoder(x_rec_ready)
        vec_rec = out_rec[0] if isinstance(out_rec, (tuple, list)) else out_rec
        if isinstance(vec_rec, dict):
            vec_rec = vec_rec.get('embedding', vec_rec.get('clipwise_output'))
        if vec_rec.ndim > 2:
            vec_rec = vec_rec.squeeze(1)
        emb_rec = F.normalize(vec_rec, p=2, dim=-1)

        sim_rec = F.cosine_similarity(emb_official, emb_rec, dim=-1).item()
        print(f"🎯 Frazione 1/{frac:02d} | Frobenius vs Nativo: {frob:6.2f} | Coseno Embedding vs Nativo: {sim_rec:.6f}")

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

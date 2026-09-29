#!/bin/bash
# Script per conversione batch da raw_wav verso raw_mp3 e raw_flac

SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"

# Percorsi base
BASE_DATASET_DIR="/leonardo/home/userexternal/lpilegg1/SEC/dataSEC/RAW_DATASET"
WAV_DIR="$BASE_DATASET_DIR/raw_wav"
MP3_DIR="$BASE_DATASET_DIR/raw_mp3"
FLAC_DIR="$BASE_DATASET_DIR/raw_flac"

if [ ! -d "$WAV_DIR" ]; then
    echo "❌ Errore: Cartella sorgente non trovata: $WAV_DIR"
    exit 1
fi

mkdir -p "$MP3_DIR" "$FLAC_DIR"

PY_CONVERT=$(mktemp /tmp/convert_dataset_XXXXXX.py)

cat << 'EOF' > "$PY_CONVERT"
import os
import sys
import glob
from concurrent.futures import ProcessPoolExecutor
import soundfile as sf
import librosa

wav_root = sys.argv[1]
mp3_root = sys.argv[2]
flac_root = sys.argv[3]

def convert_single_file(wav_path):
    rel_path = os.path.relpath(wav_path, wav_root)
    base_name = os.path.splitext(rel_path)[0]
    
    mp3_out = os.path.join(mp3_root, base_name + ".mp3")
    flac_out = os.path.join(flac_root, base_name + ".flac")
    
    # Salta se già convertiti
    need_mp3 = not os.path.exists(mp3_out)
    need_flac = not os.path.exists(flac_out)
    if not need_mp3 and not need_flac:
        return True

    os.makedirs(os.path.dirname(mp3_out), exist_ok=True)
    os.makedirs(os.path.dirname(flac_out), exist_ok=True)

    try:
        # Carica audio nativo preservando il sample rate originale
        sig, sr = sf.read(wav_path)
    except Exception:
        try:
            sig, sr = librosa.load(wav_path, sr=None, mono=False)
            if sig.ndim > 1:
                sig = sig.T
        except Exception as e:
            print(f"⚠️ Errore lettura {wav_path}: {e}", flush=True)
            return False

    # 1. Scrittura FLAC (lossless)
    if need_flac:
        try:
            sf.write(flac_out, sig, sr, format='FLAC')
        except Exception as e:
            print(f"⚠️ Errore scrittura FLAC su {flac_out}: {e}", flush=True)

    # 2. Scrittura MP3
    if need_mp3:
        try:
            # Soundfile supporta MP3 nelle versioni recenti
            sf.write(mp3_out, sig, sr, format='MP3')
        except Exception:
            # Fallback ffmpeg richiamato direttamente se libsndfile non ha il codec mp3 compilato
            cmd = f'ffmpeg -y -v error -i "{wav_path}" -vn -ar {sr} -b:a 320k "{mp3_out}"'
            ret = os.system(cmd)
            if ret != 0:
                print(f"⚠️ Fallita conversione MP3 via fallback su {mp3_out}", flush=True)

    return True

all_wavs = glob.glob(os.path.join(wav_root, "**", "*.wav"), recursive=True)
print(f"🎵 File WAV individuati: {len(all_wavs)}", flush=True)

# Esecuzione in parallelo sui core disponibili
n_workers = min(16, os.cpu_count() or 4)
print(f"⚡ Conversione avviata con {n_workers} processi paralleli...", flush=True)

with ProcessPoolExecutor(max_workers=n_workers) as executor:
    results = list(executor.map(convert_single_file, all_wavs))

print(f"✅ Conversione completata! Processati con successo: {sum(results)}/{len(all_wavs)} file.")
EOF

echo "🚀 Avvio conversione all'interno del container Singularity..."

singularity exec --no-home \
    --bind "/leonardo:/leonardo" \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "/tmp:/tmp" \
    "$SIF_FILE" \
    python3 "$PY_CONVERT" "$WAV_DIR" "$MP3_DIR" "$FLAC_DIR"

rm -f "$PY_CONVERT"

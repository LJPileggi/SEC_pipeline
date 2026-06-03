#!/bin/bash

# --- 1. SELEZIONE DEI PATH (In perfetto allineamento con la tua pipeline) ---
MY_USER=$(whoami)
BASEDIR="/leonardo_scratch/large/userexternal/${MY_USER}"
PROJECT_DIR="${BASEDIR}/SEC_pipeline"
SIF_FILE="${PROJECT_DIR}/.containers/clap_pipeline.sif"

# Path del file dei pesi di fabbrica di CLAP
export LOCAL_CLAP_WEIGHTS_PATH="${PROJECT_DIR}/.clap_weights/CLAP_weights_2023.pth"

# Path di destinazione dove convert_octave_to_msclap_mel si aspetta di trovare il file .npz
OUTPUT_DIR="${PROJECT_DIR}/.clap_weights"
OUTPUT_FILE="${OUTPUT_DIR}/clap_bn0_constants.npz"

echo "⏳ Inizializzazione script di estrazione algebrica costanti bn0 (Modalità Auto-Scansione)..."

if [ ! -f "$LOCAL_CLAP_WEIGHTS_PATH" ]; then
    echo "❌ Errore: Impossibile trovare il file dei pesi originale in: $LOCAL_CLAP_WEIGHTS_PATH"
    exit 1
fi

# --- 2. ESECUZIONE DEL MINI-SCRIPT PYTHON INLINE TRAMITE CONTAINER SINGULARITY ---
singularity exec --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$(pwd):/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 << EOF
import os
import torch
import numpy as np

pretrained_path = os.getenv("LOCAL_CLAP_WEIGHTS_PATH")
output_file = "$OUTPUT_FILE"

print(f"   📦 Apertura e scansione del dizionario dei pesi: {pretrained_path}")

# Carichiamo in modalità safe su CPU per azzerare il consumo di VRAM
state_dict = torch.load(pretrained_path, map_location='cpu')

# 🔍 STRATEGIA AGNOSTICA: Scansioniamo tutte le chiavi reali del file .pth alla ricerca di bn0
print("   🔍 Ricerca automatica delle chiavi della BatchNorm 'bn0' nel file dei pesi...")
found_keys = [k for k in state_dict.keys() if "bn0" in k]

if not found_keys:
    print("❌ Errore: Nessuna chiave contenente 'bn0' è stata trovata nel file dei pesi.")
    print("Ecco un campione delle prime 15 chiavi disponibili nel file per verifica:")
    for k in list(state_dict.keys())[:15]:
        print(f"   - {k}")
    exit(1)

# Identifichiamo il prefisso esatto basandoci su una qualsiasi delle chiavi trovate (es. escludendo la fine)
# Ad esempio, se trova 'model.audio_encoder.htsat.bn0.weight', isola il prefisso corretto.
target_sample = found_keys[0]
prefix = target_sample.split("bn0.")[0] + "bn0."
print(f"   🎯 Prefisso reale rilevato con successo: '{prefix}'")

try:
    bn0_params = {
        'running_mean': state_dict[prefix + 'running_mean'].numpy(),
        'running_var': state_dict[prefix + 'running_var'].numpy(),
        'weight': state_dict[prefix + 'weight'].numpy(),
        'bias': state_dict[prefix + 'bias'].numpy()
    }
    
    # Salvataggio in formato compresso binario nativo di NumPy
    np.savez(output_file, **bn0_params)
    print(f"✅ Conversione completata con successo!")
    print(f"   • Array estratti pronti in: {output_file}")
    print(f"   • Dimensione vettori: {bn0_params['running_mean'].shape} canali Mel.")

except KeyError as e:
    print(f"❌ Errore critico: Nonostante il prefisso, impossibile mappare la sotto-chiave specifica {e}.")
    print("Chiavi 'bn0' effettivamente presenti nel tuo file:")
    for fk in found_keys:
        print(f"   - {fk}")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo "🎉 Estrazione completata nel container. File .npz pronto."
else
    echo "❌ Errore durante l'esecuzione nel container Singularity."
    exit 1
fi

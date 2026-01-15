#!/bin/bash
# ./tests/test_lmac_pipeline.sh

# --- 1. GLOBAL ASSETS & PATHS ---
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
ROBERTA_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/roberta-base"

# Workspace temporaneo isolato su scratch
SCRATCH_TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_expl_test_$$"
CONTAINER_WORK_DIR="/tmp_data" # Mount point coerente

mkdir -p "$SCRATCH_TEMP_DIR/work_dir/roberta-base" "$SCRATCH_TEMP_DIR/work_dir/weights"
mkdir -p "$SCRATCH_TEMP_DIR/dataSEC/RAW_DATASET/raw_wav"
mkdir -p "$SCRATCH_TEMP_DIR/numba_cache"

# Mock assets per offline mode
cp -r "$ROBERTA_PATH/." "$SCRATCH_TEMP_DIR/work_dir/roberta-base/"
cp "$CLAP_WEIGHTS" "$SCRATCH_TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"

# Esportazione variabili (Usiamo i path interni al container)
export CLAP_TEXT_ENCODER_PATH="$CONTAINER_WORK_DIR/work_dir/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="$CONTAINER_WORK_DIR/work_dir/weights/CLAP_weights_2023.pth"
export NODE_TEMP_BASE_DIR="$CONTAINER_WORK_DIR/dataSEC"
export NUMBA_CACHE_DIR="$CONTAINER_WORK_DIR/numba_cache"

echo "--- 🛠️ Generazione Dati Sintetici (Sync con test_config.yaml) ---"

# 2. GENERAZIONE DATI
# Nota: Usiamo CONTAINER_WORK_DIR per i path interni allo script Python
cat << EOF > "$SCRATCH_TEMP_DIR/prepare_test.py"
import h5py
import numpy as np
import os
import sys
import torch
sys.path.append('.') 
from src.utils import HDF5EmbeddingDatasetsManager, get_config_from_yaml

def create_mock_data():
    classes, _, _, _, sr, _, _, _, _, _, _ = get_config_from_yaml("test_config.yaml")
    first_class = classes[0]
    
    # Path con cut_secs INTERO (3 invece di 3.0)
    base_path = "$CONTAINER_WORK_DIR/dataSEC/PREPROCESSED_DATASET/wav/1_octave/3_secs"
    os.makedirs(base_path, exist_ok=True)
    h5_path = os.path.join(base_path, "combined_valid.h5")
    
    manager = HDF5EmbeddingDatasetsManager(h5_path, mode='a', partitions={'splits'})
    manager.initialize_hdf5(
        embedding_dim=1024, spec_shape=(27, 256), audio_format='wav',
        cut_secs=3, n_octave=1, sample_rate=sr, seed=1, 
        noise_perc=0.3, split='valid'
    )
    
    ids = []
    for i in range(10):
        emb_id = f"0_{i}_0_1_{i}"
        ids.append(emb_id)
        manager.add_to_data_buffer(np.random.randn(1024), np.abs(np.random.randn(27, 256)), 
                                 emb_id, f"track_{i}", first_class, "sub_1")
    manager.close()
    
    with open("$CONTAINER_WORK_DIR/test_ids.txt", "w") as f:
        for eid in ids: f.write(f"{eid}\n")

    raw_path = f"$CONTAINER_WORK_DIR/dataSEC/RAW_DATASET/raw_wav/{first_class}_wav_dataset.h5"
    with h5py.File(raw_path, 'w') as f:
        f.attrs['class_idx'] = 0
        f.create_dataset('audio_wav', data=np.random.randn(20, sr*5))
        dt = np.dtype([('track_name', 'S100'), ('subclass', 'S100')])
        meta = np.array([(f'track_{i}'.encode('utf-8'), b'sub_1') for i in range(20)], dtype=dt)
        f.create_dataset('metadata_wav', data=meta)

    from src.models import FinetunedModel
    model = FinetunedModel(classes=classes, device='cpu')
    torch.save(model.state_dict(), "$CONTAINER_WORK_DIR/work_dir/weights/mock_classifier.pt")

if __name__ == '__main__':
    create_mock_data()
EOF

# Eseguiamo usando il path interno al container per lo script
singularity exec --bind "$SCRATCH_TEMP_DIR:$CONTAINER_WORK_DIR" "$SIF_FILE" \
    python3 "$CONTAINER_WORK_DIR/prepare_test.py"

echo "--- 🚀 Lancio Pipeline di Explainability ---"

# 3. ESECUZIONE PIPELINE
singularity exec --nv \
    --bind "$SCRATCH_TEMP_DIR:$CONTAINER_WORK_DIR" \
    --bind "$(pwd):/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 scripts/LMAC_pipeline.py \
        --ids_file "$CONTAINER_WORK_DIR/test_ids.txt" \
        --config_file "test_config.yaml" \
        --audio_format "wav" \
        --n_octave 1 \
        --cut_secs 3 \
        --weights_path "$CONTAINER_WORK_DIR/work_dir/weights/mock_classifier.pt" \
        --split "valid"

echo "--- 🔍 Validazione Consistenza Risultati ---"

# 4. VERIFICA INTEGRITÀ DATI
# Il path di verifica deve puntare alla cartella interna del container
RESULT_PATH_INTERNAL="$CONTAINER_WORK_DIR/dataSEC/results/explainability/wav/1_octave/3_secs"

cat << EOF > "$SCRATCH_TEMP_DIR/verify_integrity.py"
import json
import os
import sys
import numpy as np
import soundfile as sf
sys.path.append('.') 

def verify():
    summary_path = "$RESULT_PATH_INTERNAL/explainability_summary.json"
    if not os.path.exists(summary_path):
        print(f"❌ Errore: File summary non trovato in {summary_path}")
        exit(1)
        
    with open(summary_path, 'r') as f:
        data = json.load(f)
        
    for entry in data:
        # Verifica che la chiave 'audio_path' esista prima di usarla
        if 'audio_path' not in entry:
            print(f"❌ Errore: Chiave 'audio_path' mancante nel JSON per ID {entry.get('id')}")
            exit(1)
            
        # Costruzione del percorso per il controllo di ampiezza
        audio_file = os.path.join("$RESULT_PATH_INTERNAL/interpretations", os.path.basename(entry['audio_path']))
        audio, _ = sf.read(audio_file)
        if np.max(np.abs(audio)) < 1e-7:
             print(f"⚠️ Avviso: Audio silenzioso per {entry['id']}")

    print(f"✅ Test superato: {len(data)} interpretazioni verificate.")

if __name__ == '__main__':
    verify()
EOF

singularity exec --bind "$SCRATCH_TEMP_DIR:$CONTAINER_WORK_DIR" "$SIF_FILE" \
    python3 "$CONTAINER_WORK_DIR/verify_integrity.py"

# 5. CLEANUP
echo "🧹 Pulizia workspace: $SCRATCH_TEMP_DIR"
rm -rf "$SCRATCH_TEMP_DIR"

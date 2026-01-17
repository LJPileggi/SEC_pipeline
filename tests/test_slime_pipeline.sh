#!/bin/bash
# ./tests/test_slime_pipeline.sh

# --- 1. GLOBAL ASSETS & PATHS ---
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
ROBERTA_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/roberta-base"

# Workspace temporaneo isolato su scratch
SCRATCH_TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_slime_test_$$"
CONTAINER_WORK_DIR="/tmp_data" 

mkdir -p "$SCRATCH_TEMP_DIR/work_dir/roberta-base" "$SCRATCH_TEMP_DIR/work_dir/weights"
mkdir -p "$SCRATCH_TEMP_DIR/dataSEC/RAW_DATASET/raw_wav"
mkdir -p "$SCRATCH_TEMP_DIR/numba_cache"

# Mock assets per offline mode
cp -r "$ROBERTA_PATH/." "$SCRATCH_TEMP_DIR/work_dir/roberta-base/"
cp "$CLAP_WEIGHTS" "$SCRATCH_TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"

# Esportazione variabili per il container
export CLAP_TEXT_ENCODER_PATH="$CONTAINER_WORK_DIR/work_dir/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="$CONTAINER_WORK_DIR/work_dir/weights/CLAP_weights_2023.pth"
export NUMBA_CACHE_DIR="$CONTAINER_WORK_DIR/numba_cache"

echo "--- 🛠️ Generazione Dati Sintetici (Sync con test_config.yaml) ---"

# 2. GENERAZIONE DATI
# Usiamo CONTAINER_WORK_DIR per coerenza con dirs_config.py nel container
cat << EOF > "$SCRATCH_TEMP_DIR/prepare_test.py"
import h5py, numpy as np, os, sys, torch
sys.path.append('.') 
from src.utils import HDF5EmbeddingDatasetsManager, get_config_from_yaml

def create_mock_data():
    classes, _, _, _, sr, _, _, _, _, _, _ = get_config_from_yaml("test_config.yaml")
    # Path con cut_secs INTERO (3)
    base_path = "$CONTAINER_WORK_DIR/dataSEC/PREPROCESSED_DATASET/wav/1_octave/3_secs"
    os.makedirs(base_path, exist_ok=True)
    h5_path = os.path.join(base_path, "combined_valid.h5")
    
    manager = HDF5EmbeddingDatasetsManager(h5_path, mode='a', partitions={'splits'})
    manager.initialize_hdf5(1024, (27, 256), 'wav', 3, 1, sr, 1, 0.3, 'valid')
    
    ids = []
    for i in range(10):
        emb_id = f"0_{i}_0_1_{i}"
        ids.append(emb_id)
        # Segnale forte per evitare pesi nulli
        spec = np.abs(np.random.randn(27, 256)) * 50.0 
        manager.add_to_data_buffer(np.random.randn(1024), spec, emb_id, f"track_{i}", classes[0], "sub_1")
    manager.close()
    
    with open("$CONTAINER_WORK_DIR/test_ids.txt", "w") as f:
        for eid in ids: f.write(f"{eid}\n")

    # Creazione dataset RAW necessario per ricostruire le tracce in SLIME
    raw_path = f"$CONTAINER_WORK_DIR/dataSEC/RAW_DATASET/raw_wav/{classes[0]}_wav_dataset.h5"
    with h5py.File(raw_path, 'w') as f:
        f.attrs['class_idx'] = 0
        f.create_dataset('audio_wav', data=np.random.randn(20, sr*5))
        dt = np.dtype([('track_name', 'S100'), ('subclass', 'S100')])
        meta = np.array([(f'track_{i}'.encode('utf-8'), b'sub_1') for i in range(20)], dtype=dt)
        f.create_dataset('metadata_wav', data=meta)

    from src.models import FinetunedModel
    torch.save(FinetunedModel(classes=classes, device='cpu').state_dict(), 
               "$CONTAINER_WORK_DIR/work_dir/weights/mock_classifier.pt")

if __name__ == '__main__':
    create_mock_data()
EOF

# Esecuzione generazione con env esplicita
singularity exec --bind "$SCRATCH_TEMP_DIR:$CONTAINER_WORK_DIR" \
    --env NODE_TEMP_BASE_DIR="$CONTAINER_WORK_DIR/dataSEC" \
    "$SIF_FILE" python3 "$CONTAINER_WORK_DIR/prepare_test.py"

echo "--- 🚀 Lancio Pipeline SLIME ---"

# 3. ESECUZIONE PIPELINE
# Usiamo --env per assicurarci che dirs_config.py legga il path corretto
singularity exec --nv \
    --bind "$SCRATCH_TEMP_DIR:$CONTAINER_WORK_DIR" \
    --bind "$(pwd):/app" \
    --env NODE_TEMP_BASE_DIR="$CONTAINER_WORK_DIR/dataSEC" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 scripts/SLIME_pipeline.py \
        --ids_file "$CONTAINER_WORK_DIR/test_ids.txt" \
        --config_file "test_config.yaml" \
        --audio_format "wav" \
        --n_octave 1 \
        --cut_secs 3 \
        --weights_path "$CONTAINER_WORK_DIR/work_dir/weights/mock_classifier.pt" \
        --n_samples 20

echo "--- 🔍 Verifica Risultati SLIME ---"

# 4. VERIFICA PRESENZA FILE
# Il path finale costruito dalla pipeline include 'SLIME'
RESULT_JSON="$SCRATCH_TEMP_DIR/dataSEC/results/explainability/SLIME/wav/1_octave/3_secs/slime_summary.json"

if [ -f "$RESULT_JSON" ]; then
    echo "✅ Successo: slime_summary.json generato in $RESULT_JSON"
    # Opzionale: stampa un pezzetto per vedere se i pesi sono non-zero
    head -n 20 "$RESULT_JSON"
else
    echo "❌ Errore: File risultati non trovato in $RESULT_JSON"
    exit 1
fi

# 5. CLEANUP
echo "🧹 Pulizia workspace: $SCRATCH_TEMP_DIR"
rm -rf "$SCRATCH_TEMP_DIR"

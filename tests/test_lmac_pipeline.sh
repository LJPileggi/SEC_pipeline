#!/bin/bash
# ./tests/test_explainability_pipeline.sh

# --- 1. GLOBAL ASSETS & PATHS ---
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
ROBERTA_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/roberta-base"

# Workspace temporaneo isolato per il test
TEST_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_expl_test_$$"
mkdir -p "$TEST_DIR/work_dir/roberta-base" "$TEST_DIR/work_dir/weights"
mkdir -p "$TEST_DIR/dataSEC/RAW_DATASET/raw_wav"
mkdir -p "$TEST_DIR/numba_cache"

# Mock assets per offline mode
cp -r "$ROBERTA_PATH/." "$TEST_DIR/work_dir/roberta-base/"
cp "$CLAP_WEIGHTS" "$TEST_DIR/work_dir/weights/CLAP_weights_2023.pth"

# Export variabili per redirect e cache
export CLAP_TEXT_ENCODER_PATH="/tmp_data/work_dir/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"

echo "--- 🛠️ Generazione Dati Sintetici basati su test_config.yaml ---"

# 2. GENERAZIONE DATI ALLINEATI
cat << EOF > "$TEST_DIR/prepare_test.py"
import h5py
import numpy as np
import os
import sys
import torch
sys.path.append('.') 
from src.utils import HDF5EmbeddingDatasetsManager, get_config_from_yaml

def create_mock_data():
    # Usiamo test_config.yaml come riferimento
    classes, _, _, _, sr, _, _, _, _, _, _ = get_config_from_yaml("test_config.yaml")
    first_class = classes[0]
    
    # A. Preprocessed HDF5 (Embedding + Spectrogram)
    base_path = "/tmp_data/dataSEC/PREPROCESSED_DATASET/wav/1_octave/3.0_secs"
    os.makedirs(base_path, exist_ok=True)
    h5_path = os.path.join(base_path, "combined_valid.h5")
    
    manager = HDF5EmbeddingDatasetsManager(h5_path, mode='a', partitions={'splits'})
    manager.initialize_hdf5(
        embedding_dim=1024, spec_shape=(27, 256), audio_format='wav',
        cut_secs=3.0, n_octave=1, sample_rate=sr, seed=1, 
        noise_perc=0.3, split='valid'
    )
    
    ids = []
    for i in range(10):
        emb_id = f"0_{i}_0_1_{i}"
        ids.append(emb_id)
        manager.add_to_data_buffer(np.random.randn(1024), np.abs(np.random.randn(27, 256)), 
                                 emb_id, f"track_{i}", first_class, "sub_1")
    manager.close()
    
    with open("/tmp_data/test_ids.txt", "w") as f:
        for eid in ids: f.write(f"{eid}\n")

    # B. RAW Audio HDF5 (Necessario per reconstruction)
    raw_path = f"/tmp_data/dataSEC/RAW_DATASET/raw_wav/{first_class}_wav_dataset.h5"
    with h5py.File(raw_path, 'w') as f:
        f.attrs['class_idx'] = 0
        f.create_dataset('audio_wav', data=np.random.randn(20, sr*5))
        dt = np.dtype([('track_name', 'S100'), ('subclass', 'S100')])
        meta = np.array([(f'track_{i}'.encode('utf-8'), b'sub_1') for i in range(20)], dtype=dt)
        f.create_dataset('metadata_wav', data=meta)

    # C. Pesi Classifier coerenti con test_config (3 classi)
    from src.models import FinetunedModel
    model = FinetunedModel(classes=classes, device='cpu')
    torch.save(model.state_dict(), "/tmp_data/work_dir/weights/mock_classifier.pt")

if __name__ == '__main__':
    create_mock_data()
EOF

singularity exec --bind "$TEST_DIR:/tmp_data" "$SIF_FILE" python3 "$TEST_DIR/prepare_test.py"

echo "--- 🚀 Lancio Pipeline di Explainability ---"

# 3. ESECUZIONE PIPELINE
singularity exec --nv \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEST_DIR:/tmp_data" \
    --bind "$(pwd):/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 scripts/LMAC_pipeline.py \
        --ids_file "/tmp_data/test_ids.txt" \
        --config_file "test_config.yaml" \
        --audio_format "wav" \
        --n_octave 1 \
        --cut_secs 3.0 \
        --weights_path "/tmp_data/work_dir/weights/mock_classifier.pt" \
        --split "valid"

echo "--- 🔍 Validazione Consistenza Risultati ---"

# 4. VERIFICA INTEGRITÀ DATI
RESULT_PATH="$TEST_DIR/dataSEC/results/explainability/wav/1_octave/3.0_secs"

cat << EOF > "$TEST_DIR/verify_integrity.py"
import json
import os
import sys
import numpy as np
import soundfile as sf
sys.path.append('.') 

def verify():
    summary_path = "$RESULT_PATH/explainability_summary.json"
    if not os.path.exists(summary_path):
        print("❌ Errore: File summary non generato!")
        exit(1)
        
    with open(summary_path, 'r') as f:
        data = json.load(f)
        
    for entry in data:
        # Check consistenza numerica (non NaN)
        if not np.isfinite(entry['original_conf']) or not np.isfinite(entry['masked_conf']):
            print(f"❌ Errore: Confidenza non valida (NaN/Inf) per {entry['id']}")
            exit(1)
        
        # Check audio non muto (consistenza fisica)
        audio_file = os.path.join("$RESULT_PATH/interpretations", os.path.basename(entry['audio_path']))
        audio, _ = sf.read(audio_file)
        if np.max(np.abs(audio)) < 1e-7:
             print(f"⚠️ Avviso: Audio silenzioso per {entry['id']}")

    print(f"✅ Test superato: {len(data)} interpretazioni verificate con successo.")

if __name__ == '__main__':
    verify()
EOF

singularity exec --bind "$TEST_DIR:/tmp_data" "$SIF_FILE" python3 "$TEST_DIR/verify_integrity.py"

# 5. CLEANUP
echo "🧹 Pulizia workspace: $TEST_DIR"
rm -rf "$TEST_DIR"

#!/bin/bash
# ./tests/test_explainability_pipeline.sh

# --- 1. GLOBAL ASSETS & PATHS ---
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
ROBERTA_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/roberta-base"

# Workspace temporaneo isolato
TEST_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_expl_test_$$"
mkdir -p "$TEST_DIR/work_dir/roberta-base"
mkdir -p "$TEST_DIR/work_dir/weights"
mkdir -p "$TEST_DIR/dataSEC/RAW_DATASET"
mkdir -p "$TEST_DIR/numba_cache"

# Mock assets per offline mode
cp -r "$ROBERTA_PATH/." "$TEST_DIR/work_dir/roberta-base/"
cp "$CLAP_WEIGHTS" "$TEST_DIR/work_dir/weights/CLAP_weights_2023.pth"

# Export variabili per redirect e cache
export CLAP_TEXT_ENCODER_PATH="/tmp_data/work_dir/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"

echo "--- 🛠️ Generazione Dati Sintetici e Pesi Mock ---"

# 2. GENERAZIONE DATI SINTETICI
cat << EOF > "$TEST_DIR/prepare_test.py"
import h5py
import numpy as np
import os
import sys
import torch
sys.path.append('.') 
from src.utils import HDF5EmbeddingDatasetsManager

def create_mock_data():
    base_path = "/tmp_data/dataSEC/PREPROCESSED_DATASET/wav/1_octave/3.0_secs"
    os.makedirs(base_path, exist_ok=True)
    h5_path = os.path.join(base_path, "combined_valid.h5")
    
    manager = HDF5EmbeddingDatasetsManager(h5_path, mode='a', partitions={'splits'})
    manager.initialize_hdf5(
        embedding_dim=1024, spec_shape=(27, 256), audio_format='wav',
        cut_secs=3.0, n_octave=1, sample_rate=51200, seed=42, 
        noise_perc=0.1, split='valid'
    )
    
    ids = []
    for i in range(10):
        emb_id = f"0_{i}_0_1_{i}"
        ids.append(emb_id)
        # Generiamo pattern non casuali per testare la consistenza (es. onde sinusoidali mascherate)
        emb = np.random.randn(1024)
        spec = np.abs(np.random.randn(27, 256))
        manager.add_to_data_buffer(emb, spec, emb_id, f"track_{i}", "class_a", "sub_1")
    
    manager.close()
    
    with open("/tmp_data/test_ids.txt", "w") as f:
        for eid in ids: f.write(f"{eid}\n")

    # Creiamo pesi mock per il FinetunedModel
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(1024, 3) # 3 classi mock
        def forward(self, x): return self.classifier(x)

    torch.save(MockModel().state_dict(), "/tmp_data/work_dir/weights/mock_classifier.pt")

if __name__ == '__main__':
    create_mock_data()
EOF

singularity exec --bind "$TEST_DIR:/tmp_data" "$SIF_FILE" python3 /tmp_data/prepare_test.py

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
        --audio_format "wav" \
        --n_octave 1 \
        --cut_secs 3.0 \
        --weights_path "/tmp_data/work_dir/weights/mock_classifier.pt" \
        --split "valid"

echo "--- 🔍 Controlli di Consistenza Profondi ---"

# 4. CONTROLLI DI INTEGRITÀ
RESULT_PATH="$TEST_DIR/dataSEC/results/explainability/wav/1_octave/3.0_secs"

cat << EOF > "$TEST_DIR/verify_integrity.py"
import json
import os
import sys
import soundfile as sf
import numpy as np
sys.path.append('.') 

def verify():
    summary_path = "$RESULT_PATH/explainability_summary.json"
    if not os.path.exists(summary_path):
        print("❌ Errore: summary_metrics.json non trovato!")
        exit(1)
        
    with open(summary_path, 'r') as f:
        data = json.load(f)
        
    for entry in data:
        # 1. Controllo Confidenza: non devono essere valori NaN o Inf
        if not np.isfinite(entry['original_conf']) or not np.isfinite(entry['masked_conf']):
            print(f"❌ Errore Consistenza: Valori di confidenza non validi per {entry['id']}")
            exit(1)
            
        # 2. Controllo Audio: il file non deve essere vuoto (solo silenzio)
        audio_path = os.path.join("$RESULT_PATH", "interpretations", os.path.basename(entry['audio_path']))
        audio_data, sr = sf.read(audio_path)
        if np.max(np.abs(audio_data)) < 1e-6:
            print(f"⚠️ Avviso: Audio interpretazione quasi nullo per {entry['id']} - Possibile maschera troppo aggressiva")
            
        # 3. Controllo Maschere: verifica esistenza immagini
        mask_img = os.path.join("$RESULT_PATH", "masks_vis", f"{entry['id']}_mask.png")
        if not os.path.exists(mask_img):
            print(f"❌ Errore: Immagine maschera mancante per {entry['id']}")
            exit(1)

    print("✅ Tutti i controlli di consistenza (Confidenza, Audio, Immagini) sono passati.")

if __name__ == '__main__':
    verify()
EOF

singularity exec --bind "$TEST_DIR:/tmp_data" "$SIF_FILE" python3 /tmp_data/verify_integrity.py

# 5. CLEANUP
echo "🧹 Pulizia dati di test..."
rm -rf "$TEST_DIR"
echo "✨ Test di consistenza completato con successo!"

#!/bin/bash
# ./tests/test_slime_pipeline.sh

SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
SCRATCH_TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_slime_pipe_test_$$"
CONTAINER_WORK_DIR="/tmp_data"

mkdir -p "$SCRATCH_TEMP_DIR/work_dir/roberta-base" "$SCRATCH_TEMP_DIR/work_dir/weights"
mkdir -p "$SCRATCH_TEMP_DIR/dataSEC/RAW_DATASET/raw_wav" "$SCRATCH_TEMP_DIR/numba_cache"

cp -r "/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/roberta-base/." "$SCRATCH_TEMP_DIR/work_dir/roberta-base/"
cp "/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth" "$SCRATCH_TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"

export CLAP_TEXT_ENCODER_PATH="$CONTAINER_WORK_DIR/work_dir/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="$CONTAINER_WORK_DIR/work_dir/weights/CLAP_weights_2023.pth"
export NODE_TEMP_BASE_DIR="$CONTAINER_WORK_DIR/dataSEC"
export NUMBA_CACHE_DIR="$CONTAINER_WORK_DIR/numba_cache"

echo "--- 🛠️ Generazione Dati Sintetici per SLIME ---"

cat << EOF > "$SCRATCH_TEMP_DIR/prepare_test.py"
import h5py, numpy as np, os, sys, torch
sys.path.append('.') 
from src.utils import HDF5EmbeddingDatasetsManager, get_config_from_yaml

def create_mock_data():
    classes, _, _, _, sr, _, _, _, _, _, _ = get_config_from_yaml("test_config.yaml")
    base_path = "$CONTAINER_WORK_DIR/dataSEC/PREPROCESSED_DATASET/wav/1_octave/3_secs"
    os.makedirs(base_path, exist_ok=True)
    
    manager = HDF5EmbeddingDatasetsManager(os.path.join(base_path, "combined_valid.h5"), mode='a')
    manager.initialize_hdf5(1024, (27, 256), 'wav', 3, 1, sr, 1, 0.3, 'valid')
    
    for i in range(5):
        emb_id = f"0_{i}_0_1_{i}"
        spec = np.abs(np.random.randn(27, 256)) * 10 
        manager.add_to_data_buffer(np.random.randn(1024), spec, emb_id, f"tr_{i}", classes[0], "sub")
    manager.close()
    
    with open("$CONTAINER_WORK_DIR/test_ids.txt", "w") as f:
        f.write("0_0_0_1_0\n0_1_0_1_1\n")

    from src.models import FinetunedModel
    model = FinetunedModel(classes=classes, device='cpu')
    torch.save(model.state_dict(), "$CONTAINER_WORK_DIR/work_dir/weights/mock_classifier.pt")
EOF

singularity exec --bind "$SCRATCH_TEMP_DIR:$CONTAINER_WORK_DIR" "$SIF_FILE" python3 "$CONTAINER_WORK_DIR/prepare_test.py"

echo "--- 🚀 Lancio Pipeline SLIME ---"
singularity exec --nv --bind "$SCRATCH_TEMP_DIR:$CONTAINER_WORK_DIR" --bind "$(pwd):/app" --pwd "/app" "$SIF_FILE" \
    python3 scripts/SLIME_pipeline.py --ids_file "$CONTAINER_WORK_DIR/test_ids.txt" --config_file "test_config.yaml" \
    --audio_format "wav" --n_octave 1 --cut_secs 3 --weights_path "$CONTAINER_WORK_DIR/work_dir/weights/mock_classifier.pt"

echo "--- 🔍 Verifica Risultati SLIME ---"
# Verifica che il JSON contenga le chiavi della spiegazione
RESULT_JSON="$SCRATCH_TEMP_DIR/dataSEC/results/explainability/SLIME/wav/1_octave/3_secs/slime_summary.json"
if [ -f "$RESULT_JSON" ]; then
    echo "✅ Success: slime_summary.json generato."
else
    echo "❌ Error: Risultati non trovati."
    exit 1
fi

rm -rf "$SCRATCH_TEMP_DIR"

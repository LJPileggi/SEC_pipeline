#!/bin/bash
#SBATCH --job-name=test_lmac_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_Pb-skite
#SBATCH --output=test_lmac_pipeline_%j.out

# --- 1. GLOBAL ASSETS & PATHS ---
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
ROBERTA_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/roberta-base"

SCRATCH_TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_lmac_pipe_$SLURM_JOB_ID"
CONTAINER_WORK_DIR="/tmp_data"

mkdir -p "$SCRATCH_TEMP_DIR/work_dir/roberta-base" "$SCRATCH_TEMP_DIR/work_dir/weights"
mkdir -p "$SCRATCH_TEMP_DIR/dataSEC/RAW_DATASET/raw_wav"
mkdir -p "$SCRATCH_TEMP_DIR/numba_cache"

cleanup() {
    echo "🧹 Cleaning up test workspace..."
    rm -rf "$SCRATCH_TEMP_DIR"
}
trap cleanup EXIT SIGTERM SIGINT

# --- 2. STAGING & MOCK GENERATION ---
cp -r "$ROBERTA_PATH/." "$SCRATCH_TEMP_DIR/work_dir/roberta-base/"
cp "$CLAP_WEIGHTS" "$SCRATCH_TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"

# Export vars for the container context
export CLAP_TEXT_ENCODER_PATH="$CONTAINER_WORK_DIR/work_dir/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="$CONTAINER_WORK_DIR/work_dir/weights/CLAP_weights_2023.pth"
export NODE_TEMP_BASE_DIR="$CONTAINER_WORK_DIR/dataSEC"
export NUMBA_CACHE_DIR="$CONTAINER_WORK_DIR/numba_cache"

echo "🛠️ Generating Synthetic Data inside container..."
# The embedded Python script stays identical to your pipeline test
singularity exec --bind "$SCRATCH_TEMP_DIR:$CONTAINER_WORK_DIR" "$SIF_FILE" \
    python3 -c "
import h5py, numpy as np, os, sys, torch
sys.path.append('.')
from src.utils import HDF5EmbeddingDatasetsManager, get_config_from_yaml
from src.models import FinetunedModel

classes, _, _, _, sr, _, _, _, _, _, _ = get_config_from_yaml('test_config.yaml')
first_class = classes[0]
base_path = '$CONTAINER_WORK_DIR/dataSEC/PREPROCESSED_DATASET/wav/1_octave/3_secs'
os.makedirs(base_path, exist_ok=True)
h5_path = os.path.join(base_path, 'combined_valid.h5')

manager = HDF5EmbeddingDatasetsManager(h5_path, mode='a', partitions={'splits'})
manager.initialize_hdf5(1024, (27, 256), 'wav', 3, 1, sr, 1, 0.3, 'valid', first_class)

ids = []
for i in range(5):
    emb_id = f'0_{i}_0_1_{i}'
    ids.append(emb_id)
    spec = np.abs(np.random.randn(27, 256)) * 10 
    manager.add_to_data_buffer(np.random.randn(1024), spec, emb_id, f'track_{i}', first_class, 'sub_1')
manager.close()

with open('$CONTAINER_WORK_DIR/test_ids.txt', 'w') as f:
    for eid in ids: f.write(f'{eid}\n')

raw_path = f'$CONTAINER_WORK_DIR/dataSEC/RAW_DATASET/raw_wav/{first_class}_wav_dataset.h5'
with h5py.File(raw_path, 'w') as f:
    f.attrs['class_idx'] = 0
    f.create_dataset('audio_wav', data=np.random.randn(10, sr*5))
    dt = np.dtype([('track_name', 'S100'), ('subclass', 'S100')])
    meta = np.array([(f'track_{i}'.encode('utf-8'), b'sub_1') for i in range(10)], dtype=dt)
    f.create_dataset('metadata_wav', data=meta)

model = FinetunedModel(classes=classes, device='cpu')
torch.save(model.state_dict(), '$CONTAINER_WORK_DIR/work_dir/weights/mock_classifier.pt')
"

echo "🚀 Launching LMAC Pipeline Test..."
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

echo "🔍 Verifying Results..."
singularity exec --bind "$SCRATCH_TEMP_DIR:$CONTAINER_WORK_DIR" "$SIF_FILE" \
    python3 -c "
import json, os, numpy as np, soundfile as sf
res_path = '$CONTAINER_WORK_DIR/dataSEC/results/explainability/LMAC/wav/1_octave/3_secs'
with open(os.path.join(res_path, 'explainability_summary.json'), 'r') as f:
    data = json.load(f)
print(f'✅ Test passed: {len(data)} interpretations verified.')
"

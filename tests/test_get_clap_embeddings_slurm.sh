#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_Pb-skite

"""
Test script for verifying the CLAP embedding pipeline on the Leonardo Cluster.
This script automates the setup of a temporary execution environment, generates 
synthetic raw data, and executes a multi-GPU embedding run using Singularity 
containers.

Workflow:
 1. Environment and path initialization;
 2. Workspace and local cache preparation;
 3. Synthetic dataset generation;
 4. Configuration of the firewall-safe environment;
 5. Parallel pipeline execution via srun;
 6. Log merging and performance analysis.

Args (Environment Variables):
 - VERBOSE (bool, default: True): Enables detailed logging for the redirector and setup;
 - BENCHMARK_CONFIG_FILE (str): YAML config defining the processing parameters;
 - BENCHMARK_AUDIO_FORMAT (str): Audio format to process (e.g., 'wav');
 - BENCHMARK_N_OCTAVE (str): Octave band resolution for spectrograms.
"""

# 🎯 VERBOSITY CONTROL
# Set to 'True' for full setup diagnostics, 'False' for production milestones only.
export VERBOSE=True

# --- 1. GLOBAL VARIABLES AND PATHS ---
# Define paths for the Singularity image, model weights, and job-specific storage.
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"

TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_job_$SLURM_JOB_ID"
PERSISTENT_DESTINATION="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/benchmark_logs/$SLURM_JOB_ID"

BENCHMARK_CONFIG_FILE="test_config.yaml" 
BENCHMARK_AUDIO_FORMAT="wav"
BENCHMARK_N_OCTAVE="1"

# --- 2. ENVIRONMENT PREPARATION ---
# Initialize temporary job directories.
mkdir -p "$TEMP_DIR/dataSEC/RAW_DATASET"
mkdir -p "$TEMP_DIR/work_dir"

# 🎯 RIGID CACHE STRUCTURE (HuggingFace Style)
# msclap expects weights in a specific directory hierarchy for offline loading.
CACHE_BASE="$TEMP_DIR/work_dir/huggingface/hub/models--microsoft--msclap/snapshots/main"
mkdir -p "$CACHE_BASE"
cp "$CLAP_SCRATCH_WEIGHTS" "$CACHE_BASE/CLAP_weights_2023.pth"

# --- 3. SYNTHETIC DATA GENERATION ---
# Create a temporary Python script to generate a fake raw audio HDF5 dataset for testing.
cat << EOF > "$TEMP_DIR/work_dir/create_h5_data.py"
import sys, os
sys.path.append('/app')
from tests.utils.create_fake_raw_audio_h5 import create_fake_raw_audio_h5 
base_dir = os.environ.get('NODE_TEMP_BASE_DIR', '/tmp_data/dataSEC')
TARGET_DIR = os.path.join(base_dir, 'RAW_DATASET') 
if __name__ == '__main__':
    create_fake_raw_audio_h5(TARGET_DIR)
EOF

export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
# Execute generator within Singularity container
singularity exec --bind "$TEMP_DIR:/tmp_data" --bind "$(pwd):/app" --pwd "/app" "$SIF_FILE" \
    python3 "/tmp_data/work_dir/create_h5_data.py"

# --- 4. PIPELINE ENVIRONMENT CONFIGURATION ---
# Set variables for offline HuggingFace operations and local weight redirection.
export HF_HOME="$TEMP_DIR/work_dir/huggingface"
export HF_HUB_OFFLINE=1 

# Copy RoBERTa text encoder assets to the temporary job directory
mkdir -p "$TEMP_DIR/roberta-base"
cp -r /leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/roberta-base/. "$TEMP_DIR/roberta-base/"

export CLAP_TEXT_ENCODER_PATH="/tmp_data/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/CLAP_weights_2023.pth"
cp "$CLAP_SCRATCH_WEIGHTS" "$TEMP_DIR/work_dir/CLAP_weights_2023.pth"

export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC" 
export NO_EMBEDDING_SAVE="True" # Disable HDF5 saving for performance benchmarking

# System-level optimizations for Leonardo Cluster
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

if [ "$VERBOSE" = "True" ]; then
    echo "--- PHYSICAL FILE VERIFICATION ---"
    ls -l "$TEMP_DIR/roberta-base"
    echo "LOCAL_CLAP_WEIGHTS_PATH: $LOCAL_CLAP_WEIGHTS_PATH"
    echo "CLAP_TEXT_ENCODER_PATH: $CLAP_TEXT_ENCODER_PATH"
    echo "---------------------------"
fi

# --- 5. PIPELINE EXECUTION ---
# Launch parallel workers using srun. Each task is mapped to a dedicated GPU.
echo "🚀 Starting Parallel Embedding Pipeline..."

srun --unbuffered -l -n 4 --export=ALL --cpu-bind=none \
    singularity exec --nv \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEMP_DIR:/tmp_data" \
    --bind "$(pwd):/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 scripts/get_clap_embeddings.py \
        --config_file "$BENCHMARK_CONFIG_FILE" \
        --n_octave "$BENCHMARK_N_OCTAVE" \
        --audio_format "$BENCHMARK_AUDIO_FORMAT"

# --- 6. SEQUENTIAL LOG MERGING ---
# Consolidate individual rank logs into unified JSON files for each segment duration.
echo "🔗 Merging rank-specific logs..."
cat << EOF > "$TEMP_DIR/work_dir/join_logs_wrapper.py"
import sys, os
sys.path.append('/app')
from src.utils import join_logs
from src.dirs_config import basedir_preprocessed
def main():
    base_path = os.path.join(basedir_preprocessed, "$BENCHMARK_AUDIO_FORMAT", "${BENCHMARK_N_OCTAVE}_octave")
    if not os.path.exists(base_path): return
    for entry in os.listdir(base_path):
        target_dir = os.path.join(base_path, entry)
        if os.path.isdir(target_dir) and entry.endswith("_secs"):
            print(f"Merging: {entry}")
            join_logs(target_dir)
if __name__ == '__main__': main()
EOF

singularity exec --bind "$TEMP_DIR:/tmp_data" --bind "$(pwd):/app" --pwd "/app" "$SIF_FILE" \
    python3 "/tmp_data/work_dir/join_logs_wrapper.py"

# --- 7. FINAL ANALYSIS ---
# Calculate and print performance statistics (timing, throughput).
echo "📊 Running Execution Time Analysis..."
cat << EOF > "$TEMP_DIR/work_dir/analysis_wrapper.py"
import os, sys
sys.path.append('/app') 
import tests.utils.analyse_test_execution_times as analysis_module
analysis_module.config_test_folder = os.path.join(os.getenv('NODE_TEMP_BASE_DIR'), 'PREPROCESSED_DATASET')
if __name__ == '__main__':
    results = analysis_module.analyze_execution_times("$BENCHMARK_AUDIO_FORMAT", "$BENCHMARK_N_OCTAVE", "$BENCHMARK_CONFIG_FILE")
    analysis_module.print_analysis_results(results)
EOF

singularity exec --bind "$TEMP_DIR:/tmp_data" --bind "$(pwd):/app" --pwd "/app" "$SIF_FILE" \
    python3 "/tmp_data/work_dir/analysis_wrapper.py"

# --- 8. RESULTS PERSISTENCE AND CLEANUP ---
# Save the preprocessed data to persistent scratch and notify completion.
mkdir -p "$PERSISTENT_DESTINATION"
cp -r "$TEMP_DIR/dataSEC" "$PERSISTENT_DESTINATION/"
rm -rf "$TEMP_DIR" # Uncomment to enable cleanup of temporary job data
echo "✅ Job Completed. Results stored in $PERSISTENT_DESTINATION"

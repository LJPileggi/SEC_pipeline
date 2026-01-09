#!/bin/bash
#
"""
Interactive and local test script for the CLAP embedding pipeline.
Designed to be launched on an already allocated compute node (e.g., via 'salloc') 
or a local workstation. It manages temporary data on high-performance scratch 
storage and performs automated cleanup.

Key Features:
 1. Dynamic path binding for Singularity containers;
 2. Local redirection of CLAP weights to bypass internet checks;
 3. Inline generation of HDF5 test datasets;
 4. Automated merging of rank logs and execution time analysis.

Args (Environment Variables):
 - VERBOSE (bool, default: True): Toggles detailed diagnostic logs;
 - SCRATCH_TEMP_DIR (path): Temporary base directory on scratch storage ($$ suffix for uniqueness).
"""

# 🎯 VERBOSITY CONTROL
# Set to 'True' for detailed firewall redirect logs, 'False' for production milestones.
export VERBOSE=True

# --- 1. GLOBAL VARIABLES AND PATHS ---
# Define core assets and temporary storage structure
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"

# Create a unique temporary directory on scratch using the script's PID ($$)
SCRATCH_TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_data_pipeline_$$"
CONTAINER_SCRATCH_BASE="/scratch_base" 
CONTAINER_WORK_DIR="/app/temp_work"

TEMP_DIR="$SCRATCH_TEMP_DIR/work_dir" 
TEMP_PYTHON_SCRIPT_PATH="$TEMP_DIR/create_h5_data.py"
TEMP_ANALYSE_WRAPPER_PATH="$TEMP_DIR/analysis_main_wrapper.py"
TEMP_JOIN_LOGS_PATH="$TEMP_DIR/join_logs_wrapper.py"

# Benchmark specific configurations
BENCHMARK_CONFIG_FILE="test_config.yaml" 
BENCHMARK_AUDIO_FORMAT="wav"
BENCHMARK_N_OCTAVE="1"

# --- 2. DATA PREPARATION ON SCRATCH ---
# Setting up the temporary job workspace
echo "--- 🛠️ Preparing Temporary Data on Scratch ($SCRATCH_TEMP_DIR) ---"
mkdir -p "$TEMP_DIR" 
mkdir -p "$SCRATCH_TEMP_DIR/dataSEC/RAW_DATASET" 

# Copying CLAP weights to the temporary workspace
CLAP_LOCAL_WEIGHTS="$TEMP_DIR/CLAP_weights_2023.pth"
cp "$CLAP_SCRATCH_WEIGHTS" "$CLAP_LOCAL_WEIGHTS" 

# --- 3. EXECUTION CONFIGURATION ---
# Define internal container paths for weight redirection
echo "--- ⚙️ Configuring Execution Environment ---"
export CLAP_TEXT_ENCODER_PATH="/usr/local/clap_cache/tokenizer_model/" 
export LOCAL_CLAP_WEIGHTS_PATH="$CONTAINER_WORK_DIR/CLAP_weights_2023.pth"
export NO_EMBEDDING_SAVE="True" # Benchmark mode: do not save embeddings to disk
export NODE_TEMP_BASE_DIR="$CONTAINER_SCRATCH_BASE/dataSEC"

# --- 4. DYNAMIC HDF5 DATASET GENERATION ---
# Generates a fake raw audio HDF5 for testing purposes
echo "Generating dynamic HDF5 test dataset..."
cat << EOF > "$TEMP_PYTHON_SCRIPT_PATH"
import sys, os
sys.path.append('.')
from tests.utils.create_fake_raw_audio_h5 import create_fake_raw_audio_h5 
TARGET_DIR = os.path.join(os.getenv('NODE_TEMP_BASE_DIR'), 'RAW_DATASET') 
if __name__ == '__main__':
    create_fake_raw_audio_h5(TARGET_DIR)
EOF

singularity exec --bind "$TEMP_DIR:$CONTAINER_WORK_DIR" --bind "$SCRATCH_TEMP_DIR:$CONTAINER_SCRATCH_BASE" \
    "$SIF_FILE" python3 "$CONTAINER_WORK_DIR/create_h5_data.py"

# --- 5. PIPELINE EXECUTION ---
# Launching the main embedding script
echo "--- 🚀 Launching Interactive Execution ---\n"
singularity exec \
    --bind "$TEMP_DIR:$CONTAINER_WORK_DIR" \
    --bind "$(pwd)/configs:/app/configs" \
    --bind "$SCRATCH_TEMP_DIR:$CONTAINER_SCRATCH_BASE" \
    "$SIF_FILE" \
    python3 scripts/get_clap_embeddings.py \
        --config_file "$BENCHMARK_CONFIG_FILE" \
        --n_octave "$BENCHMARK_N_OCTAVE" \
        --audio_format "$BENCHMARK_AUDIO_FORMAT"

# --- 5.5. SEQUENTIAL LOG MERGING ---
# Combines rank-specific logs into unified files
echo "--- 🔗 Running Sequential join_logs ---"
cat << EOF > "$TEMP_JOIN_LOGS_PATH"
import sys, os
sys.path.append('.')
from src.utils import join_logs
from src.dirs_config import basedir_preprocessed
def main():
    base_path = os.path.join(basedir_preprocessed, "$BENCHMARK_AUDIO_FORMAT", "${BENCHMARK_N_OCTAVE}_octave")
    if not os.path.exists(base_path): return
    for entry in os.listdir(base_path):
        target_dir = os.path.join(base_path, entry)
        if os.path.isdir(target_dir) and entry.endswith("_secs"):
            join_logs(target_dir)
if __name__ == '__main__': main()
EOF

singularity exec --bind "$TEMP_DIR:$CONTAINER_WORK_DIR" --bind "$SCRATCH_TEMP_DIR:$CONTAINER_SCRATCH_BASE" \
    "$SIF_FILE" python3 "$CONTAINER_WORK_DIR/join_logs_wrapper.py"

# --- 6. FINAL PERFORMANCE ANALYSIS ---
# Aggregates timing data and prints the final benchmark report
echo "--- 📊 Starting Time Analysis ---"
cat << EOF > "$TEMP_ANALYSE_WRAPPER_PATH"
import os, sys, argparse
sys.path.append('.') 
import tests.utils.analyse_test_execution_times as analysis_module
analysis_module.config_test_folder = os.path.join(os.getenv('NODE_TEMP_BASE_DIR'), 'PREPROCESSED_DATASET')
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--n_octave', type=str, required=True)
    parser.add_argument('--audio_format', type=str, required=True)
    args = parser.parse_args()
    results = analysis_module.analyze_execution_times(args.audio_format, args.n_octave, args.config_file)
    analysis_module.print_analysis_results(results)
if __name__ == '__main__': main()
EOF

singularity exec \
    --bind "$TEMP_DIR:$CONTAINER_WORK_DIR" \
    --bind "$SCRATCH_TEMP_DIR:$CONTAINER_SCRATCH_BASE" \
    --env NODE_TEMP_BASE_DIR="$CONTAINER_SCRATCH_BASE/dataSEC" \
    "$SIF_FILE" \
    python3 "$CONTAINER_WORK_DIR/analysis_main_wrapper.py" \
    --config_file "$BENCHMARK_CONFIG_FILE" \
    --n_octave "$BENCHMARK_N_OCTAVE" \
    --audio_format "$BENCHMARK_AUDIO_FORMAT"

# --- 7. FINAL CLEANUP ---
# Deletes all temporary HDF5 files, logs, and weights from scratch
echo "Cleaning up temporary scratch data: $SCRATCH_TEMP_DIR"
rm -rf "$SCRATCH_TEMP_DIR"
echo "Execution and Analysis Complete."

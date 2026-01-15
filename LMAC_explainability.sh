#!/bin/bash
# ./run_explainability.sh

# --- 1. PARAMETERS CHECK ---
if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <ids_file> <audio_format> <n_octave> <cut_secs> <weights_path> [split]"
    echo "Example: $0 my_samples.txt wav 1 3.0 path/to/classifier.pt valid"
    exit 1
fi

IDS_FILE=$1
FORMAT=$2
OCTAVE=$3
SECS=$4
WEIGHTS=$5
SPLIT=${6:-"valid"} # Default to valid if not specified

# --- 2. GLOBAL ASSETS ---
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
ROBERTA_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/roberta-base"

# Unique workspace for this manual run
WORK_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_expl_$$"
mkdir -p "$WORK_DIR/roberta-base"
mkdir -p "$WORK_DIR/weights"
mkdir -p "$WORK_DIR/numba_cache"

# --- 3. ENVIRONMENT SETUP ---
# Copy assets to scratch for offline use
cp -r "$ROBERTA_PATH/." "$WORK_DIR/roberta-base/"
cp "$CLAP_WEIGHTS" "$WORK_DIR/weights/CLAP_weights_2023.pth"

# Redirects for offline patching and Numba cache
export CLAP_TEXT_ENCODER_PATH="/tmp_data/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/weights/CLAP_weights_2023.pth"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export VERBOSE=False # Set to True for detailed firewall logs

echo "🎬 Starting Explainability Pipeline for $FORMAT - Octave $OCTAVE - $SECS secs..."

# --- 4. EXECUTION ---
singularity exec --nv \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$WORK_DIR:/tmp_data" \
    --bind "$(pwd):/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 scripts/LMAC_pipeline.py \
        --ids_file "$IDS_FILE" \
        --audio_format "$FORMAT" \
        --n_octave "$OCTAVE" \
        --cut_secs "$SECS" \
        --weights_path "$WEIGHTS" \
        --split "$SPLIT"

# --- 5. CLEANUP ---
rm -rf "$WORK_DIR"
echo "✅ Finished. Check your results in the PREPROCESSED_DATASET folder."

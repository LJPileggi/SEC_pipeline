#!/bin/bash
# ./SLIME_explainability.sh

# --- 1. PARAMETERS ---
if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <ids_file> <audio_format> <n_octave> <cut_secs> <weights_path> [expl_type]"
    echo "Example: $0 target_ids.txt wav 1 3 path/to/weights.pt time_frequency"
    exit 1
fi

IDS_FILE=$1
FORMAT=$2
OCTAVE=$3
SECS=$4
WEIGHTS=$5
EXPL_TYPE=${6:-"time_frequency"} # Default a time_frequency

# --- 2. ASSETS ---
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
ROBERTA_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/roberta-base"

WORK_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_slime_$$"
mkdir -p "$WORK_DIR/roberta-base" "$WORK_DIR/weights" "$WORK_DIR/numba_cache"

cp -r "$ROBERTA_PATH/." "$WORK_DIR/roberta-base/"
cp "$CLAP_WEIGHTS" "$WORK_DIR/weights/CLAP_weights_2023.pth"

# environment setup
export CLAP_TEXT_ENCODER_PATH="/tmp_data/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/weights/CLAP_weights_2023.pth"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"

echo "🎬 Starting SLIME Pipeline ($EXPL_TYPE) - $FORMAT - $SECS secs..."

# --- 3. EXECUTION ---
singularity exec --nv \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$WORK_DIR:/tmp_data" \
    --bind "$(pwd):/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 scripts/SLIME_pipeline.py \
        --ids_file "$IDS_FILE" \
        --audio_format "$FORMAT" \
        --n_octave "$OCTAVE" \
        --cut_secs "$SECS" \
        --weights_path "$WEIGHTS" \
        --expl_type "$EXPL_TYPE"

rm -rf "$WORK_DIR"
echo "✅ SLIME Completed."

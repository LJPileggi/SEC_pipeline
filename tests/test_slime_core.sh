#!/bin/bash
# ./tests/test_slime_core.sh

SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
ROBERTA_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/roberta-base"

TEST_WORK_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_slime_core_test_$$"
mkdir -p "$TEST_WORK_DIR/roberta-base" "$TEST_WORK_DIR/weights" "$TEST_WORK_DIR/numba_cache"

cp -r "$ROBERTA_PATH/." "$TEST_WORK_DIR/roberta-base/"
cp "$CLAP_WEIGHTS" "$TEST_WORK_DIR/weights/CLAP_weights_2023.pth"

export TEST_WEIGHTS_PATH="/tmp_data/weights/dummy_classifier.pt"
export CLAP_TEXT_ENCODER_PATH="/tmp_data/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/weights/CLAP_weights_2023.pth"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"

echo "🚀 Launching SLIME Core Logic Test..."
singularity exec --nv \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEST_WORK_DIR:/tmp_data" \
    --bind "$(pwd):/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 tests/utils/test_slime_logic.py

rm -rf "$TEST_WORK_DIR"

#!/bin/bash
# ./tests/test_lmac_core.sh

# --- CONFIGURATION ---
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
ROBERTA_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/roberta-base"

# Unique workspace for testing
TEST_WORK_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_lmac_test_$$"
mkdir -p "$TEST_WORK_DIR/roberta-base"
mkdir -p "$TEST_WORK_DIR/weights"

# Mock weights file for testing FinetunedModel loading
# In a real scenario, this would be your .pt checkpoint
DUMMY_WEIGHTS="$TEST_WORK_DIR/weights/dummy_classifier.pt"

cp -r "$ROBERTA_PATH/." "$TEST_WORK_DIR/roberta-base/"
cp "$CLAP_WEIGHTS" "$TEST_WORK_DIR/weights/CLAP_weights_2023.pth"

# Environment variables for Offline Patching
export CLAP_TEXT_ENCODER_PATH="/tmp_data/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/weights/CLAP_weights_2023.pth"
export TEST_WEIGHTS_PATH="/tmp_data/weights/dummy_classifier.pt"
export VERBOSE=True

echo "🚀 Launching Full L-MAC Functional Test..."

singularity exec --nv \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEST_WORK_DIR:/tmp_data" \
    --bind "$(pwd):/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 tests/utils/test_lmac_logic.py

# Cleanup
rm -rf "$TEST_WORK_DIR"

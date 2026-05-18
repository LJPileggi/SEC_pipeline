#!/bin/bash
#SBATCH --job-name=EMB_COMP
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=64GB
#SBATCH --account=IscrC_Pb-skite
#SBATCH --output=%x_%j.out

# --- CONFIGURAZIONE ---
PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"

# Percorsi dei file H5 generati precedentemente
AUDIO_H5="/leonardo_scratch/large/userexternal/$USER/dataSEC/PREPROCESSED_DATASET/wav/0_octave/7_secs/combined_test.h5"
OCTAVE_H5="/leonardo_scratch/large/userexternal/$USER/dataSEC/PREPROCESSED_DATASET/wav/3_octave/7_secs/combined_test.h5"
NO_INJECT_DIR="/leonardo_scratch/large/userexternal/$USER/dataSEC/PREPROCESSED_DATASET/wav/3_octave_no_inject/7_secs/combined_test.h5"

OUTPUT_DIR="/leonardo_scratch/large/userexternal/$USER/EMBEDDING_COMPARISON"
mkdir -p "$OUTPUT_DIR"

SIF_FILE="$PROJECT_DIR/.containers/clap_pipeline.sif"

echo "🚀 Avvio confronto embeddings su nodo di calcolo..."

singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$(pwd):/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 alia/distrib_comparison.py \
        --audio_h5 "$AUDIO_H5" \
        --octave_h5 "$OCTAVE_H5" \
        --no_inject_h5 "$NO_INJECT_DIR" \
        --output_dir "$OUTPUT_DIR"

echo "✅ Completato. Risultati in $OUTPUT_DIR"

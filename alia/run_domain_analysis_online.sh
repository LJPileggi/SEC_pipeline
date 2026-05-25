#!/bin/bash

MY_USER=$(whoami)
BASEDIR="/leonardo_scratch/large/userexternal/${MY_USER}"
PROJECT_DIR="${BASEDIR}/SEC_pipeline"
DATASEC_DIR="${BASEDIR}/dataSEC"
SIF_FILE="${PROJECT_DIR}/.containers/clap_pipeline.sif"
CONFIG_FILE="${PROJECT_DIR}/configs/config0.yaml"

# Vincoli atomici per il nodo di login
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="" # Esclude totalmente CUDA per evitare eccezioni

echo "⏳ Esecuzione Analisi di Dominio Online (50 campioni/classe)..."

singularity exec --no-home \
    --bind "${BASEDIR}:/app" \
    --bind "${PROJECT_DIR}:/app/${PROJECT_DIR}" \
    --bind "${DATASEC_DIR}:/app/${DATASEC_DIR}" \
    --pwd "/app/${PROJECT_DIR}" \
    "$SIF_FILE" \
    python3 alia/evaluate_domain_distance_online.py \
        --config_file "$CONFIG_FILE" \
        --n_octave "3" \
        --audio_format "wav \
        --cut_secs 7 \
        --samples_per_class 50

echo "✅ Analisi conclusa."

#!/bin/bash

# Setup script for the CLAP environment on Cineca.
# Execute ONLY ONCE on the login node.

# --- 1. VARIABLES AND PATHS ---
echo "--- 1. VARIABLES AND PATHS ---"
USER_AREA="$1"
PROJECT_ROOT_DIR="$USER_AREA/SEC_pipeline" 

CLAP_WEIGHTS_DIR="$PROJECT_ROOT_DIR/.clap_weights"
CLAP_WEIGHTS_FILE="CLAP_weights_2023.pth"
CLAP_WEIGHTS_PATH="${CLAP_WEIGHTS_DIR}/${CLAP_WEIGHTS_FILE}"
CLAP_WEIGHTS_URL="https://huggingface.co/microsoft/msclap/resolve/main/CLAP_weights_2023.pth"

# 🎯 Assets for TextEncoder (RoBERTa)
ROBERTA_DIR="$CLAP_WEIGHTS_DIR/roberta-base"

CONTAINER_DIR="$PROJECT_ROOT_DIR/.containers"
SIF_PATH="$CONTAINER_DIR/clap_pipeline.sif"

# --- 2. DIRECTORY CREATION ---
echo "--- 2. DIRECTORY CREATION ---"
mkdir -p "$CLAP_WEIGHTS_DIR"
mkdir -p "$ROBERTA_DIR" 
mkdir -p "$CONTAINER_DIR"
mkdir -p "$USER_AREA/dataSEC/PREPROCESSED_DATA"

# --- 3. DOWNLOAD WEIGHTS (.PTH) ---
echo "--- 3. DOWNLOAD WEIGHTS ---"
if [ ! -f "$CLAP_WEIGHTS_PATH" ]; then
    echo "Weights not found. Starting download..."
    wget -O "$CLAP_WEIGHTS_PATH" "$CLAP_WEIGHTS_URL"
else
    echo "CLAP weights already present."
fi

# 🎯 Download RoBERTa assets (config and vocab)
echo "--- 4. DOWNLOAD ROBERTA ASSETS ---"
if [ ! -f "$ROBERTA_DIR/config.json" ]; then
    echo "Downloading RoBERTa assets..."
    wget -P "$ROBERTA_DIR" https://huggingface.co/roberta-base/resolve/main/config.json
    wget -P "$ROBERTA_DIR" https://huggingface.co/roberta-base/resolve/main/vocab.json
    wget -P "$ROBERTA_DIR" https://huggingface.co/roberta-base/resolve/main/merges.txt
else
    echo "RoBERTa assets already present."
fi

echo "========================================================================================="
echo "Partial setup completed."
echo "IMPORTANT: The 'clap_pipeline.sif' file must be transferred manually to:"
echo "$CONTAINER_DIR"
echo "Use Cineca native methods (Data Mover or scp). Refer to official documentation:"
echo "https://docs.hpc.cineca.it/hpc/hpc_data_storage.html"
echo "========================================================================================="

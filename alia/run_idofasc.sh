#!/bin/bash
#SBATCH --job-name=IDOFASC_HPC
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=128GB
#SBATCH --account=IscrC_Pb-skite
#SBATCH --output=%x_%j.out

FORMAT=${1:-wav}

# --- CONFIGURAZIONE PERCORSI (ESTRATTI DAI TUOI SCRIPT) ---
PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
# Dove il tuo convertitore SALVA i file h5
H5_SOURCE_DIR="/leonardo_scratch/large/userexternal/$USER/dataSEC/RAW_DATASET/raw_$FORMAT"
# Cartella parallela per i risultati
OUTPUT_DIR="/leonardo_scratch/large/userexternal/$USER/IDOFASC_results_$FORMAT"

SIF_FILE="$PROJECT_DIR/.containers/clap_pipeline.sif"
LOCAL_TMP="${local_scratch:-/leonardo_scratch/large/userexternal/$USER/tmp_job_$SLURM_JOB_ID}"

echo "📦 Preparazione nodi e Stage-in..."
mkdir -p "$LOCAL_TMP/data" "$LOCAL_TMP/weights" "$LOCAL_TMP/models" "$LOCAL_TMP/numba_cache"
mkdir -p "$OUTPUT_DIR"

# COPIA MIRATA: prendiamo solo i file del formato richiesto (es. *_wav_dataset.h5)
echo "📂 Copia file h5 da $H5_SOURCE_DIR..."
cp "$H5_SOURCE_DIR"/*_"${FORMAT}"_dataset.h5 "$LOCAL_TMP/data/" 2>/dev/null

# Verifica immediata della copia nello script shell
if [ $(ls "$LOCAL_TMP/data/" | wc -l) -eq 0 ]; then
    echo "❌ ERRORE CRITICO: Nessun file trovato in $H5_SOURCE_DIR con pattern *_${FORMAT}_dataset.h5"
    exit 1
fi

cp "$PROJECT_DIR/.clap_weights/CLAP_weights_2023.pth" "$LOCAL_TMP/weights/"
cp -r "$PROJECT_DIR/.clap_weights/roberta-base/." "$LOCAL_TMP/models/"

export AUDIO_FORMAT="$FORMAT"
export CLAP_TEXT_ENCODER_PATH="/tmp_data/models"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/weights/CLAP_weights_2023.pth"
export INPUT_HDF5_DIR="/tmp_data/data"
export OUTPUT_RESULTS_PATH="/output_dir"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export HF_HUB_OFFLINE=1

echo "🚀 Esecuzione Python..."
singularity exec --nv --no-home \
    --bind "$LOCAL_TMP:/tmp_data" \
    --bind "$PROJECT_DIR:/app" \
    --bind "$OUTPUT_DIR:/output_dir" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 alia/IDOFASC_HPC.py

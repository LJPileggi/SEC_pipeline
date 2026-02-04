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

# Formato audio dinamico (passato come variabile o default a wav)
FORMAT=${1:-wav}

# --- CONFIGURAZIONE PERCORSI ---
PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
H5_SOURCE_DIR="/leonardo/home/userexternal/$USER/SEC/dataSEC/RAW_DATASET/raw_$FORMAT"
SIF_FILE="$PROJECT_DIR/.containers/clap_pipeline.sif"

# Uso del local_scratch per evitare Permission Denied
LOCAL_TMP="${local_scratch:-/leonardo_scratch/large/userexternal/$USER/tmp_job_$SLURM_JOB_ID}"

echo "📦 Stage-in dei dati ($FORMAT) su $LOCAL_TMP..."
mkdir -p "$LOCAL_TMP/data" "$LOCAL_TMP/weights" "$LOCAL_TMP/models"

cp "$H5_SOURCE_DIR"/*.h5 "$LOCAL_TMP/data/" 2>/dev/null
cp "$PROJECT_DIR/.clap_weights/CLAP_weights_2023.pth" "$LOCAL_TMP/weights/"
cp -r "$PROJECT_DIR/.clap_weights/roberta-base/." "$LOCAL_TMP/models/"

# EXPORT VARIABILI
export AUDIO_FORMAT="$FORMAT"
export CLAP_TEXT_ENCODER_PATH="$LOCAL_TMP/models"
export LOCAL_CLAP_WEIGHTS_PATH="$LOCAL_TMP/weights/CLAP_weights_2023.pth"
export INPUT_HDF5_DIR="$LOCAL_TMP/data"
export OUTPUT_RESULTS_PATH="$PROJECT_DIR/IDOFASC_results_$FORMAT"
export HF_HUB_OFFLINE=1

echo "🚀 Avvio IDOFASC_HPC.py tramite Singularity per formato: $FORMAT"
singularity exec --nv --no-home \
    --bind "$LOCAL_TMP:/tmp_data" \
    --bind "$PROJECT_DIR:/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 alia/IDOFASC_HPC.py

#!/bin/bash
#SBATCH --job-name=IDOFASC_prod
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

# --- CONFIGURAZIONE PERCORSI ---
PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
H5_SOURCE_DIR="/leonardo/home/userexternal/$USER/SEC/dataSEC/RAW_DATASET/raw_wav"
LOCAL_TMP="/tmp_data/$SLURM_JOB_ID"

# 1. STAGE-IN (Copia locale per evitare latenza Lustre)
echo "📦 Stage-in dei dati su $LOCAL_TMP..."
mkdir -p "$LOCAL_TMP/data" "$LOCAL_TMP/weights" "$LOCAL_TMP/models"

cp "$H5_SOURCE_DIR"/*.h5 "$LOCAL_TMP/data/"
cp "$PROJECT_DIR/.clap_weights/CLAP_weights_2023.pth" "$LOCAL_TMP/weights/"
cp -r "$PROJECT_DIR/.clap_weights/roberta-base" "$LOCAL_TMP/models/"

# 2. EXPORT VARIABILI PER LE PATCH
export CLAP_TEXT_ENCODER_PATH="$LOCAL_TMP/models/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="$LOCAL_TMP/weights/CLAP_weights_2023.pth"
export INPUT_HDF5_DIR="$LOCAL_TMP/data"
export OUTPUT_RESULTS_PATH="$PROJECT_DIR/IDOFASC_results"
export HF_HUB_OFFLINE=1

# 3. ESECUZIONE
echo "🚀 Avvio IDOFASC_HPC.py tramite Singularity..."
singularity exec --nv --no-home \
    --bind "$LOCAL_TMP:/tmp_data" \
    --bind "$PROJECT_DIR:/app" \
    --pwd "/app" \
    "$PROJECT_DIR/.containers/clap_pipeline.sif" \
    python3 alia/IDOFASC_HPC.py

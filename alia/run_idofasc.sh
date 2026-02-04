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

# Formato audio dinamico
FORMAT=${1:-wav}

# --- CONFIGURAZIONE PERCORSI ---
# Il progetto è in /.../SEC_pipeline
PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
# I risultati vanno in /.../IDOFASC_results_$FORMAT (parallela a SEC_pipeline)
PARENT_DIR="/leonardo_scratch/large/userexternal/$USER"
OUTPUT_DIR="$PARENT_DIR/IDOFASC_results_$FORMAT"

H5_SOURCE_DIR="/leonardo/home/userexternal/$USER/SEC/dataSEC/RAW_DATASET/raw_$FORMAT"
SIF_FILE="$PROJECT_DIR/.containers/clap_pipeline.sif"

# Uso del local_scratch per lo stage-in
LOCAL_TMP="${local_scratch:-/leonardo_scratch/large/userexternal/$USER/tmp_job_$SLURM_JOB_ID}"

echo "📦 Stage-in dei dati ($FORMAT) su $LOCAL_TMP..."
mkdir -p "$LOCAL_TMP/data" "$LOCAL_TMP/weights" "$LOCAL_TMP/models"
mkdir -p "$OUTPUT_DIR"

cp "$H5_SOURCE_DIR"/*.h5 "$LOCAL_TMP/data/" 2>/dev/null
cp "$PROJECT_DIR/.clap_weights/CLAP_weights_2023.pth" "$LOCAL_TMP/weights/"
cp -r "$PROJECT_DIR/.clap_weights/roberta-base/." "$LOCAL_TMP/models/"

# EXPORT VARIABILI PER IL PYTHON
export AUDIO_FORMAT="$FORMAT"
export CLAP_TEXT_ENCODER_PATH="/tmp_data/models"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/weights/CLAP_weights_2023.pth"
export INPUT_HDF5_DIR="/tmp_data/data"
export OUTPUT_RESULTS_PATH="/output_dir"
export HF_HUB_OFFLINE=1
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"

echo "🚀 Avvio IDOFASC_HPC.py per formato: $FORMAT"
# Bindiamo la cartella di output esterna al container
singularity exec --nv --no-home \
    --bind "$LOCAL_TMP:/tmp_data" \
    --bind "$PROJECT_DIR:/app" \
    --bind "$OUTPUT_DIR:/output_dir" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 alia/IDOFASC_HPC.py

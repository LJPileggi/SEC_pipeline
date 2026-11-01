#!/bin/bash
#SBATCH --job-name=val_full_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_BrISkite_0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

VAE_EPOCH=${1:-75}
REFINER_EPOCH=${2:-25}
TASK_MODE=${3:-"both"}

PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_val_full_$SLURM_JOB_ID"
SIF_FILE="$PROJECT_DIR/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="$PROJECT_DIR/.clap_weights/CLAP_weights_2023.pth"
CLAP_BN0_CONSTANTS="$PROJECT_DIR/.clap_weights/clap_bn0_constants.npz"
DATASEC_GLOBAL="/leonardo_scratch/large/userexternal/$USER/dataSEC"

VAE_MODELS_DIR="$PROJECT_DIR/.models/vae_model"
REFINER_MODELS_DIR="$PROJECT_DIR/.models/refiner_model"

# Risoluzione robusta checkpoint VAE
VAE_FILE=""
if [ "$VAE_EPOCH" == "latest" ]; then
    VAE_FILE=$(ls "$VAE_MODELS_DIR"/dual_vae_epoch_*.pt 2>/dev/null | awk -F'_' '{print $NF, $0}' | sort -n | tail -n 1 | cut -d' ' -f2-)
else
    VAE_IDX=$(( VAE_EPOCH - 1 ))
    if [ -f "$VAE_MODELS_DIR/dual_vae_epoch_${VAE_IDX}.pt" ]; then
        VAE_FILE="$VAE_MODELS_DIR/dual_vae_epoch_${VAE_IDX}.pt"
    elif [ -f "$VAE_MODELS_DIR/dual_vae_epoch_${VAE_EPOCH}.pt" ]; then
        VAE_FILE="$VAE_MODELS_DIR/dual_vae_epoch_${VAE_EPOCH}.pt"
    else
        VAE_FILE=$(ls "$VAE_MODELS_DIR"/dual_vae_epoch_*.pt 2>/dev/null | awk -F'_' '{print $NF, $0}' | sort -n | tail -n 1 | cut -d' ' -f2-)
    fi
fi

if [ -z "$VAE_FILE" ] || [ ! -f "$VAE_FILE" ]; then
    echo "❌ [ERROR] Checkpoint VAE non trovato in $VAE_MODELS_DIR"
    exit 1
fi

# Risoluzione robusta checkpoint Refiner
REF_FILE=""
if [ "$REFINER_EPOCH" == "latest" ]; then
    REF_FILE=$(ls "$REFINER_MODELS_DIR"/refiner_epoch_*.pt 2>/dev/null | awk -F'_' '{print $NF, $0}' | sort -n | tail -n 1 | cut -d' ' -f2-)
else
    REF_IDX=$(( REFINER_EPOCH - 1 ))
    if [ -f "$REFINER_MODELS_DIR/refiner_epoch_${REF_IDX}.pt" ]; then
        REF_FILE="$REFINER_MODELS_DIR/refiner_epoch_${REF_IDX}.pt"
    elif [ -f "$REFINER_MODELS_DIR/refiner_epoch_${REFINER_EPOCH}.pt" ]; then
        REF_FILE="$REFINER_MODELS_DIR/refiner_epoch_${REFINER_EPOCH}.pt"
    else
        REF_FILE=$(ls "$REFINER_MODELS_DIR"/refiner_epoch_*.pt 2>/dev/null | awk -F'_' '{print $NF, $0}' | sort -n | tail -n 1 | cut -d' ' -f2-)
    fi
fi

if [ -z "$REF_FILE" ] || [ ! -f "$REF_FILE" ]; then
    echo "❌ [ERROR] Checkpoint Refiner non trovato in $REFINER_MODELS_DIR"
    exit 1
fi

mkdir -p "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav"
mkdir -p "$TEMP_DIR/work_dir/weights"
mkdir -p "$TEMP_DIR/numba_cache"
mkdir -p "$TEMP_DIR/staged_models"

cleanup_job_scratch() {
    trap - SIGTERM SIGINT
    echo "🧹 Purging temporary scratch space..."
    rm -rf "$TEMP_DIR"
    exit 0
}
trap 'cleanup_job_scratch' SIGTERM SIGINT

echo "=========================================================="
echo "🎯 Full Pipeline Validation Launcher"
echo "   • Mode:               $TASK_MODE"
echo "   • Single VAE Staged:  $VAE_FILE"
echo "   • Single Ref Staged:  $REF_FILE"
echo "=========================================================="

echo "📦 Stage-in: Staging dataset, pesi CLAP e singoli modelli..."
cp "$CLAP_SCRATCH_WEIGHTS" "$TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"
[ -f "$CLAP_BN0_CONSTANTS" ] && cp "$CLAP_BN0_CONSTANTS" "$TEMP_DIR/work_dir/weights/clap_bn0_constants.npz"
cp "$DATASEC_GLOBAL/RAW_DATASET/raw_wav"/*.h5 "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav/" 2>/dev/null

cp "$VAE_FILE" "$TEMP_DIR/staged_models/target_vae.pt"
cp "$REF_FILE" "$TEMP_DIR/staged_models/target_refiner.pt"

export BASEDIR="$TEMP_DIR"
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
export LOCAL_CLAP_BN0_CONSTANTS_PATH="/tmp_data/work_dir/weights/clap_bn0_constants.npz"
export LOCAL_VAE_CHECKPOINT="/tmp_data/staged_models/target_vae.pt"
export LOCAL_REFINER_CHECKPOINT="/tmp_data/staged_models/target_refiner.pt"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export PYTHONUNBUFFERED=1

if [ "$TASK_MODE" == "both" ] || [ "$TASK_MODE" == "metrics" ]; then
    echo "🔬 [1/2] Calcolo Metriche Quantitative End-to-End..."
    singularity exec --nv --no-home \
        --bind "/leonardo_scratch:/leonardo_scratch" \
        --bind "$TEMP_DIR:/tmp_data" \
        --bind "$PROJECT_DIR:/app" --pwd "/app" \
        "$SIF_FILE" \
        python3 src/filterbank_refiner/pipeline/validate_full_pipeline.py \
            --vae_checkpoint "$LOCAL_VAE_CHECKPOINT" \
            --refiner_checkpoint "$LOCAL_REFINER_CHECKPOINT"
fi

if [ "$TASK_MODE" == "both" ] || [ "$TASK_MODE" == "plot" ]; then
    echo "🎨 [2/2] Generazione Grafici Diagnostici a 7 Pannelli..."
    singularity exec --nv --no-home \
        --bind "/leonardo_scratch:/leonardo_scratch" \
        --bind "$TEMP_DIR:/tmp_data" \
        --bind "$PROJECT_DIR:/app" --pwd "/app" \
        "$SIF_FILE" \
        python3 src/filterbank_refiner/pipeline/plot_full_pipeline.py \
            --vae_checkpoint "$LOCAL_VAE_CHECKPOINT" \
            --refiner_checkpoint "$LOCAL_REFINER_CHECKPOINT" \
            --target_fraction 3
fi

cleanup_job_scratch

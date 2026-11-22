#!/bin/bash
#SBATCH --job-name=mel_refiner_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_BrISkite_0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --signal=B:TERM@60

NUM_EPOCHS=${1:-25}
VAE_TARGET_EPOCH=${2:-75}
START_EPOCH_ARG=${3:-0}
TOTAL_PLANNED_EPOCHS=${4:-25}

PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_train_refiner_$SLURM_JOB_ID"
SIF_FILE="$PROJECT_DIR/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="$PROJECT_DIR/.clap_weights/CLAP_weights_2023.pth"
CLAP_BN0_CONSTANTS="$PROJECT_DIR/.clap_weights/clap_bn0_constants.npz"
DATASEC_GLOBAL="/leonardo_scratch/large/userexternal/$USER/dataSEC"
MODELS_GLOBAL="$PROJECT_DIR/.models/vae_model"
REFINER_GLOBAL="$PROJECT_DIR/.models/refiner_model"

mkdir -p "$REFINER_GLOBAL"

# Risoluzione START_EPOCH (supporta numero esplicito oppure "auto")
if [ "$START_EPOCH_ARG" == "auto" ]; then
    LATEST_REFINER=$(ls "$REFINER_GLOBAL"/refiner_epoch_*.pt 2>/dev/null | sed -E 's/.*refiner_epoch_([0-9]+)\.pt/\1 &/' | sort -n | tail -n 1)
    if [ -n "$LATEST_REFINER" ]; then
        LAST_EP=$(echo "$LATEST_REFINER" | awk '{print $1}')
        START_EPOCH=$((LAST_EP + 1))
    else
        START_EPOCH=0
    fi
else
    START_EPOCH=$START_EPOCH_ARG
fi

# Verifica preventiva esistenza checkpoint di resume se START_EPOCH > 0
if [ "$START_EPOCH" -gt 0 ]; then
    PREV_EP=$((START_EPOCH - 1))
    if [ ! -f "$REFINER_GLOBAL/refiner_epoch_${PREV_EP}.pt" ]; then
        echo "❌ [ERROR] Checkpoint di resume non trovato: $REFINER_GLOBAL/refiner_epoch_${PREV_EP}.pt"
        exit 1
    fi
fi

# Risoluzione robusta del checkpoint VAE target
TARGET_FILE=""
if [ "$VAE_TARGET_EPOCH" == "latest" ]; then
    TARGET_FILE=$(ls "$MODELS_GLOBAL"/dual_vae_epoch_*.pt 2>/dev/null | awk -F'_' '{print $NF, $0}' | sort -n | tail -n 1 | cut -d' ' -f2-)
else
    VAE_IDX=$((VAE_TARGET_EPOCH - 1))
    if [ -f "$MODELS_GLOBAL/dual_vae_epoch_${VAE_IDX}.pt" ]; then
        TARGET_FILE="$MODELS_GLOBAL/dual_vae_epoch_${VAE_IDX}.pt"
    elif [ -f "$MODELS_GLOBAL/dual_vae_epoch_${VAE_TARGET_EPOCH}.pt" ]; then
        TARGET_FILE="$MODELS_GLOBAL/dual_vae_epoch_${VAE_TARGET_EPOCH}.pt"
    else
        TARGET_FILE=$(ls "$MODELS_GLOBAL"/dual_vae_epoch_*.pt 2>/dev/null | awk -F'_' '{print $NF, $0}' | sort -n | tail -n 1 | cut -d' ' -f2-)
    fi
fi

if [ -z "$TARGET_FILE" ] || [ ! -f "$TARGET_FILE" ]; then
    echo "❌ [ERROR] Nessun checkpoint VAE valido trovato in $MODELS_GLOBAL!"
    exit 1
fi

mkdir -p "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav"
mkdir -p "$TEMP_DIR/work_dir/weights"
mkdir -p "$TEMP_DIR/numba_cache"
mkdir -p "$TEMP_DIR/single_vae_model"

CLEANED_UP=0

cleanup_job_scratch() {
    local sig=$1
    if [ "$CLEANED_UP" -eq 1 ]; then
        return
    fi
    CLEANED_UP=1

    trap - SIGTERM SIGINT SIGHUP EXIT
    echo -e "\n⚠️ Signal [${sig:-EXIT}] intercettato! Avvio consolidamento I/O..."

    if [ -n "$SRUN_PID" ] && kill -0 "$SRUN_PID" 2>/dev/null; then
        echo "🛑 Arresto controllato dello step srun (PID: $SRUN_PID)..."
        kill -SIGTERM "$SRUN_PID" 2>/dev/null
        sleep 3
    fi

    sync

    if [ -d "$TEMP_DIR" ]; then
        echo "🧹 Rimozione directory temporanea $TEMP_DIR..."
        rm -rf "$TEMP_DIR"
    fi

    echo "🏁 Procedura di completamento terminata."
    exit 0
}

trap 'cleanup_job_scratch SIGTERM' SIGTERM
trap 'cleanup_job_scratch SIGINT'  SIGINT
trap 'cleanup_job_scratch SIGHUP'  SIGHUP
trap 'cleanup_job_scratch EXIT'    EXIT

echo "=========================================================="
echo "🎯 Refiner Training Launcher"
echo "   • Start Epoch:        $START_EPOCH"
echo "   • Num Epochs (Run):   $NUM_EPOCHS"
echo "   • Total Planned Ep.:  $TOTAL_PLANNED_EPOCHS"
echo "   • VAE Target Epoch:   $VAE_TARGET_EPOCH"
echo "   • VAE Checkpoint:     $TARGET_FILE"
echo "   • Refiner Save Dir:   $REFINER_GLOBAL"
echo "=========================================================="

echo "📦 Stage-in: Staging datasets, pesi CLAP e SINGOLO checkpoint VAE..."
cp "$CLAP_SCRATCH_WEIGHTS" "$TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"
[ -f "$CLAP_BN0_CONSTANTS" ] && cp "$CLAP_BN0_CONSTANTS" "$TEMP_DIR/work_dir/weights/clap_bn0_constants.npz"
cp "$DATASEC_GLOBAL/RAW_DATASET/raw_wav"/*.h5 "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav/" 2>/dev/null
cp "$TARGET_FILE" "$TEMP_DIR/single_vae_model/target_dual_vae.pt"

export BASEDIR="$TEMP_DIR"
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
export LOCAL_CLAP_BN0_CONSTANTS_PATH="/tmp_data/work_dir/weights/clap_bn0_constants.npz"
export LOCAL_VAE_CHECKPOINT="/tmp_data/single_vae_model/target_dual_vae.pt"
export REFINER_CHECKPOINT_DIR="/app/.models/refiner_model"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(expr 22000 + ${SLURM_JOB_ID} % 10000)

echo "🚀 Launching Refiner Training on 4 ranks..."

srun --unbuffered -l -n 4 --export=ALL --cpu-bind=none \
    singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEMP_DIR:/tmp_data" \
    --bind "$PROJECT_DIR:/app" --pwd "/app" \
    "$SIF_FILE" \
    python3 src/filterbank_refiner/pipeline/train.py \
        --start_epoch "$START_EPOCH" \
        --epochs "$NUM_EPOCHS" \
        --total_planned_epochs "$TOTAL_PLANNED_EPOCHS" \
        --vae_checkpoint "/tmp_data/single_vae_model/target_dual_vae.pt" &

SRUN_PID=$!
wait "$SRUN_PID"
SRUN_STATUS=$?

exit $SRUN_STATUS

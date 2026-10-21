#!/bin/bash
#SBATCH --job-name=unet_train_cross
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=13:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_BrISkite_0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --signal=B:TERM@60

START_EPOCH=${1:-0}
NUM_EPOCHS=${2:-125}

PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_train_diff_$SLURM_JOB_ID"
SIF_FILE="$PROJECT_DIR/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="$PROJECT_DIR/.clap_weights/CLAP_weights_2023.pth"
CLAP_BN0_CONSTANTS="$PROJECT_DIR/.clap_weights/clap_bn0_constants.npz"
DATASEC_GLOBAL="/leonardo_scratch/large/userexternal/$USER/dataSEC"

export MODELS_GLOBAL="$PROJECT_DIR/.models/diff_model"

mkdir -p "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav"
mkdir -p "$TEMP_DIR/work_dir/weights"
mkdir -p "$TEMP_DIR/numba_cache"
mkdir -p "$TEMP_DIR/models/diff_model"
mkdir -p "$MODELS_GLOBAL"

CLEANED_UP=0

cleanup_job_scratch() {
    local sig=$1
    if [ "$CLEANED_UP" -eq 1 ]; then
        return
    fi
    CLEANED_UP=1

    trap - SIGTERM SIGINT SIGHUP EXIT
    echo -e "\n⚠️ Signal [${sig:-EXIT}] intercettato! Avvio sincronizzazione e pulizia scratch..."

    # Arresto controllato del processo srun se ancora attivo in background
    if [ -n "$SRUN_PID" ] && kill -0 "$SRUN_PID" 2>/dev/null; then
        echo "🛑 Arresto controllato dello step srun (PID: $SRUN_PID)..."
        kill -SIGTERM "$SRUN_PID" 2>/dev/null
        sleep 3
    fi

    # Stage-out atomico dei checkpoint della diffusione
    if [ -d "$TEMP_DIR/models/diff_model" ]; then
        echo "📦 Stage-out: Sincronizzazione checkpoint diffusione su $MODELS_GLOBAL..."
        rsync -rlt --ignore-errors "$TEMP_DIR/models/diff_model/" "$MODELS_GLOBAL/"
        sync
        echo "✅ Checkpoint diffusione sincronizzati con successo."
    fi

    if [ -d "$TEMP_DIR" ]; then
        echo "🧹 Rimozione directory temporanea $TEMP_DIR..."
        rm -rf "$TEMP_DIR"
    fi

    echo "🏁 Procedura di uscita completata."
    exit 0
}

trap 'cleanup_job_scratch SIGTERM' SIGTERM
trap 'cleanup_job_scratch SIGINT'  SIGINT
trap 'cleanup_job_scratch SIGHUP'  SIGHUP
trap 'cleanup_job_scratch EXIT'    EXIT

echo "=========================================================="
echo "🎯 Diffusion Training (Guided Cross-Attention):"
echo "   • Start Epoch:      $START_EPOCH"
echo "   • Number of Epochs: $NUM_EPOCHS"
echo "   • End Epoch Target: $((START_EPOCH + NUM_EPOCHS - 1))"
echo "=========================================================="

echo "📦 Stage-in: Moving global WAV HDF5 datasets and weights..."
cp "$CLAP_SCRATCH_WEIGHTS" "$TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"
[ -f "$CLAP_BN0_CONSTANTS" ] && cp "$CLAP_BN0_CONSTANTS" "$TEMP_DIR/work_dir/weights/clap_bn0_constants.npz"
cp "$DATASEC_GLOBAL/RAW_DATASET/raw_wav"/*.h5 "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav/" 2>/dev/null

if [ -d "$MODELS_GLOBAL" ]; then
    echo "📂 Stage-in dei checkpoint precedenti da $MODELS_GLOBAL..."
    cp -r "$MODELS_GLOBAL/." "$TEMP_DIR/models/diff_model/" 2>/dev/null
fi

export BASEDIR="$TEMP_DIR"
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
export LOCAL_CLAP_BN0_CONSTANTS_PATH="/tmp_data/work_dir/weights/clap_bn0_constants.npz"
export MODEL_CHECKPOINT_DIR="/tmp_data/models/diff_model"

export INJECT_OCTAVE="True"
export VERBOSE="False"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(expr 20000 + ${SLURM_JOB_ID} % 10000)

echo "🚀 Launching Distributed Training on 4 ranks..."

# Esecuzione in background con monitoraggio del PID per intercettare i segnali
srun --unbuffered -l -n 4 --export=ALL --cpu-bind=none \
    singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEMP_DIR:/tmp_data" \
    --bind "$PROJECT_DIR:/app" --pwd "/app" \
    "$SIF_FILE" \
    python3 src/filterbank_diffusion/pipeline/train.py \
        --start_epoch "$START_EPOCH" \
        --epochs "$NUM_EPOCHS" &

SRUN_PID=$!
wait "$SRUN_PID"
SRUN_STATUS=$?

exit $SRUN_STATUS

#!/bin/bash
#SBATCH --job-name=dual_vae_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_BrISkite_0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --signal=B:TERM@60

START_EPOCH=${1:-0}
NUM_EPOCHS=${2:-50}
BOTTLENECK_TYPE=${3:-"rvq_2d"}

PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_train_vae_$SLURM_JOB_ID"
SIF_FILE="$PROJECT_DIR/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="$PROJECT_DIR/.clap_weights/CLAP_weights_2023.pth"
CLAP_BN0_CONSTANTS="$PROJECT_DIR/.clap_weights/clap_bn0_constants.npz"
DATASEC_GLOBAL="/leonardo_scratch/large/userexternal/$USER/dataSEC"

export MODELS_GLOBAL="$PROJECT_DIR/.models/vae_model"

mkdir -p "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav"
mkdir -p "$TEMP_DIR/work_dir/weights"
mkdir -p "$TEMP_DIR/numba_cache"
mkdir -p "$TEMP_DIR/models/vae_model"
mkdir -p "$MODELS_GLOBAL"

# Variabile per evitare esecuzioni ricorsive o doppie del cleanup
CLEANED_UP=0

cleanup_job_scratch() {
    local sig=$1
    if [ "$CLEANED_UP" -eq 1 ]; then
        return
    fi
    CLEANED_UP=1
    
    # Rimuoviamo i trap per evitare loop durante l'uscita
    trap - SIGTERM SIGINT SIGHUP EXIT
    
    echo -e "\n⚠️ Signal [${sig:-EXIT}] intercettato! Avvio procedura di salvataggio e stage-out..."
    
    # 1. Se il processo srun è ancora in background, fermalo con grazia
    if [ -n "$SRUN_PID" ] && kill -0 "$SRUN_PID" 2>/dev/null; then
        echo "🛑 Arresto controllato dello step srun (PID: $SRUN_PID)..."
        kill -SIGTERM "$SRUN_PID" 2>/dev/null
        # Breve attesa per permettere a Singularity di liberare le risorse I/O
        sleep 3
    fi

    # 2. Stage-out prioritario dei checkpoint generati prima di toccare la cartella
    if [ -d "$TEMP_DIR/models/vae_model" ]; then
        echo "📦 Stage-out: Sincronizzazione atomica dei checkpoint su $MODELS_GLOBAL..."
        rsync -rlt --ignore-errors "$TEMP_DIR/models/vae_model/" "$MODELS_GLOBAL/"
        sync
        echo "✅ Checkpoint salvati con successo in $MODELS_GLOBAL."
    fi

    # 3. Rimozione della cartella temporanea locale
    if [ -d "$TEMP_DIR" ]; then
        echo "🧹 Rimozione directory temporanea $TEMP_DIR..."
        rm -rf "$TEMP_DIR"
    fi

    echo "🏁 Procedura di uscita completata."
    exit 0
}

# Trap attivo su tutti i segnali di stop inviati da scancel o timeout Slurm
trap 'cleanup_job_scratch SIGTERM' SIGTERM
trap 'cleanup_job_scratch SIGINT'  SIGINT
trap 'cleanup_job_scratch SIGHUP'  SIGHUP
trap 'cleanup_job_scratch EXIT'    EXIT

echo "=========================================================="
echo "🎯 Dual-VAE Training Config:"
echo "   • Start Epoch:      $START_EPOCH"
echo "   • Number of Epochs: $NUM_EPOCHS"
echo "   • Bottleneck Type:  $BOTTLENECK_TYPE"
echo "=========================================================="

echo "📦 Stage-in: Moving global WAV HDF5 datasets and weights..."
cp "$CLAP_SCRATCH_WEIGHTS" "$TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"
[ -f "$CLAP_BN0_CONSTANTS" ] && cp "$CLAP_BN0_CONSTANTS" "$TEMP_DIR/work_dir/weights/clap_bn0_constants.npz"
cp "$DATASEC_GLOBAL/RAW_DATASET/raw_wav"/*.h5 "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav/" 2>/dev/null

if [ -d "$MODELS_GLOBAL" ]; then
    echo "📂 Stage-in dei checkpoint VAE precedenti..."
    cp -r "$MODELS_GLOBAL/." "$TEMP_DIR/models/vae_model/" 2>/dev/null
fi

export BASEDIR="$TEMP_DIR"
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
export LOCAL_CLAP_BN0_CONSTANTS_PATH="/tmp_data/work_dir/weights/clap_bn0_constants.npz"
export MODEL_CHECKPOINT_DIR="/tmp_data/models/vae_model"
export BOTTLENECK_TYPE="$BOTTLENECK_TYPE"

export INJECT_OCTAVE="True"
export VERBOSE="False"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(expr 20000 + ${SLURM_JOB_ID} % 10000)

echo "🚀 Launching Distributed Training on 4 ranks..."

# Esecuzione in background con cattura del PID per non mascherare i segnali
srun --unbuffered -l -n 4 --export=ALL --cpu-bind=none \
    singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEMP_DIR:/tmp_data" \
    --bind "$PROJECT_DIR:/app" --pwd "/app" \
    "$SIF_FILE" \
    python3 src/filterbank_vae/pipeline/train.py \
        --start_epoch "$START_EPOCH" \
        --epochs "$NUM_EPOCHS" \
        --bottleneck_type "$BOTTLENECK_TYPE" &

SRUN_PID=$!
wait "$SRUN_PID"
SRUN_STATUS=$?

# Il cleanup finale su EXIT si occuperà automaticamente dello stage-out
exit $SRUN_STATUS

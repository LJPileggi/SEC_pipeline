#!/bin/bash
#SBATCH --job-name=vae_lat_diag
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_BrISkite_0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --signal=B:TERM@60

FRACTION=${1:-32}
SAMPLES_PER_CLASS=${2:-50}

PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_diag_vae_$SLURM_JOB_ID"
SIF_FILE="$PROJECT_DIR/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="$PROJECT_DIR/.clap_weights/CLAP_weights_2023.pth"
CLAP_BN0_CONSTANTS="$PROJECT_DIR/.clap_weights/clap_bn0_constants.npz"
DATASEC_GLOBAL="/leonardo_scratch/large/userexternal/$USER/dataSEC"
MODELS_GLOBAL="$PROJECT_DIR/.models/vae_model"
RESULTS_GLOBAL="$PROJECT_DIR/results/vae_latent_diagnostics"

mkdir -p "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav"
mkdir -p "$TEMP_DIR/work_dir/weights"
mkdir -p "$TEMP_DIR/numba_cache"
mkdir -p "$TEMP_DIR/models/vae_model"
mkdir -p "$RESULTS_GLOBAL"

CLEANED_UP=0

cleanup_job_scratch() {
    local sig=$1
    if [ "$CLEANED_UP" -eq 1 ]; then
        return
    fi
    CLEANED_UP=1

    trap - SIGTERM SIGINT SIGHUP EXIT
    echo -e "\n⚠️ Signal [${sig:-EXIT}] intercettato! Salvataggio risultati diagnostici..."

    if [ -n "$SRUN_PID" ] && kill -0 "$SRUN_PID" 2>/dev/null; then
        echo "🛑 Arresto controllato dello step diagnostico..."
        kill -SIGTERM "$SRUN_PID" 2>/dev/null
        sleep 2
    fi

    # Stage-out prioritario delle cartelle di risultato
    if [ -d "$PROJECT_DIR/results/vae_latent_diagnostics" ]; then
        echo "📦 Stage-out: Sincronizzazione risultati su Lustre..."
        sync
    fi

    if [ -d "$TEMP_DIR" ]; then
        echo "🧹 Rimozione scratch temporaneo $TEMP_DIR..."
        rm -rf "$TEMP_DIR"
    fi

    echo "🏁 Conclusione completata."
    exit 0
}

trap 'cleanup_job_scratch SIGTERM' SIGTERM
trap 'cleanup_job_scratch SIGINT'  SIGINT
trap 'cleanup_job_scratch SIGHUP'  SIGHUP
trap 'cleanup_job_scratch EXIT'    EXIT

echo "=========================================================="
echo "🎯 Dual-VAE Latent & Codebook Deep Diagnostics"
echo "   • Octave Fraction:   1/$FRACTION"
echo "   • Samples per Class: $SAMPLES_PER_CLASS"
echo "   • Global Models:     $MODELS_GLOBAL"
echo "=========================================================="

echo "📦 Stage-in: Pesi CLAP, dataset RAW e checkpoint..."
cp "$CLAP_SCRATCH_WEIGHTS" "$TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"
[ -f "$CLAP_BN0_CONSTANTS" ] && cp "$CLAP_BN0_CONSTANTS" "$TEMP_DIR/work_dir/weights/clap_bn0_constants.npz"
cp "$DATASEC_GLOBAL/RAW_DATASET/raw_wav"/*.h5 "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav/" 2>/dev/null

LATEST_VAE=$(ls -v "$MODELS_GLOBAL"/dual_vae_epoch_*.pt 2>/dev/null | tail -n 1)
if [ -z "$LATEST_VAE" ]; then
    echo "❌ [ERROR] Nessun checkpoint VAE trovato in $MODELS_GLOBAL!"
    exit 1
fi
echo "📦 Checkpoint selezionato: $LATEST_VAE"
cp "$LATEST_VAE" "$TEMP_DIR/models/vae_model/target_dual_vae.pt"

export BASEDIR="$TEMP_DIR"
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
export LOCAL_CLAP_BN0_CONSTANTS_PATH="/tmp_data/work_dir/weights/clap_bn0_constants.npz"
export CHECKPOINT_PATH="/tmp_data/models/vae_model/target_dual_vae.pt"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "🚀 Lancio Diagnosi Latenti su GPU..."
srun --unbuffered -l -n 1 --export=ALL --cpu-bind=none \
    singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEMP_DIR:/tmp_data" \
    --bind "$PROJECT_DIR:/app" --pwd "/app" \
    "$SIF_FILE" \
    python3 src/filterbank_vae/pipeline/diagnose_latents.py \
        --fraction "$FRACTION" \
        --samples_per_class "$SAMPLES_PER_CLASS" \
        --checkpoint "/tmp_data/models/vae_model/target_dual_vae.pt" &

SRUN_PID=$!
wait "$SRUN_PID"
SRUN_STATUS=$?

exit $SRUN_STATUS

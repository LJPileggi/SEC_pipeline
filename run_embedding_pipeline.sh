#!/bin/bash
# run_embedding_pipeline.sh

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <config_file> <audio_format> <n_octave> <inject_octave> <mode> [world_size] [generative_backbone] [vae_epoch] [refiner_epoch]"
    echo "Modes: slurm, local"
    echo "Backbones: vae_refiner (default), vae, diffusion"
    exit 1
fi

CONFIG_FILE=$1
AUDIO_FORMAT=$2
N_OCTAVE=$3
INJECT_OCTAVE_CMD=$4
MODE=$5
WORLD_SIZE=${6:-4}
GENERATIVE_BACKBONE=${7:-vae_refiner}
VAE_TARGET_EPOCH=${8:-50}
REFINER_TARGET_EPOCH=${9:-25}

if [ "$N_OCTAVE" -eq 0 ]; then
    INJECT_OCTAVE_CMD="False"
fi

PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
SIF_FILE="$PROJECT_DIR/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="$PROJECT_DIR/.clap_weights/CLAP_weights_2023.pth"
CLAP_BN0_CONSTANTS="$PROJECT_DIR/.clap_weights/clap_bn0_constants.npz"
ROBERTA_PATH="$PROJECT_DIR/.clap_weights/roberta-base"
DATASEC_GLOBAL="/leonardo_scratch/large/userexternal/$USER/dataSEC"

DIFFUSION_MODELS_PATH="$PROJECT_DIR/.models/diff_model"
VAE_MODELS_PATH="$PROJECT_DIR/.models/vae_model"
REFINER_MODELS_PATH="$PROJECT_DIR/.models/refiner_model"

run_slurm() {
    local j_name="prod_${AUDIO_FORMAT}_oct${N_OCTAVE}_${GENERATIVE_BACKBONE}"
    local script="submit_${j_name}.sh"
    local target_folder="${N_OCTAVE}_octave"
    if [ "$N_OCTAVE" -ne 0 ] && [ "$INJECT_OCTAVE_CMD" = "False" ]; then
        target_folder="${N_OCTAVE}_octave_no_inject"
    fi
    local FINAL_DEST="$DATASEC_GLOBAL/PREPROCESSED_DATASET/$AUDIO_FORMAT/$target_folder"

    cat << EOF > "$script"
#!/bin/bash
#SBATCH --job-name=$j_name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_BrISkite_0
#SBATCH --output=%x_%j.out

export TEMP_DIR="/leonardo_scratch/large/userexternal/\$USER/tmp_job_\$SLURM_JOB_ID"
TARGET_GLOBAL="$FINAL_DEST"

mkdir -p "\$TEMP_DIR/dataSEC/RAW_DATASET/raw_$AUDIO_FORMAT"
mkdir -p "\$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$AUDIO_FORMAT/$target_folder"
mkdir -p "\$TEMP_DIR/work_dir/weights"
mkdir -p "\$TEMP_DIR/work_dir/diff_model"
mkdir -p "\$TEMP_DIR/work_dir/vae_model"
mkdir -p "\$TEMP_DIR/work_dir/refiner_model"
mkdir -p "\$TEMP_DIR/roberta-base"
mkdir -p "\$TEMP_DIR/numba_cache"

finalize_and_cleanup() {
    trap - SIGTERM SIGINT
    echo "⚠️ Signal caught! Settling I/O..."
    sleep 5
    if [ -d "\$TEMP_DIR" ]; then
        echo "🔗 Merging logs & Joining HDF5..."
        singularity exec --no-home --bind "\$TEMP_DIR:/tmp_data" --bind "\$(pwd):/app" --pwd "/app" "$SIF_FILE" \\
            python3 "/tmp_data/work_dir/join_logs_wrapper.py"
        singularity exec --nv --no-home --bind "/leonardo_scratch:/leonardo_scratch" --bind "\$TEMP_DIR:/tmp_data" \\
            --bind "\$(pwd):/app" --pwd "/app" "$SIF_FILE" \\
            python3 scripts/join_hdf5.py --config_file "$CONFIG_FILE" --n_octave "$N_OCTAVE" --audio_format "$AUDIO_FORMAT"
        echo "📦 Stage-out..."
        mkdir -p "\$TARGET_GLOBAL"
        rsync -rlt "\$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$AUDIO_FORMAT/$target_folder/" "\$TARGET_GLOBAL/"
        rm -rf "\$TEMP_DIR"
    fi
    exit 0
}
trap 'finalize_and_cleanup' SIGTERM SIGINT

echo "📦 Stage-in CLAP weights and RAW datasets..."
cp "$CLAP_SCRATCH_WEIGHTS" "\$TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"
[ -f "$CLAP_BN0_CONSTANTS" ] && cp "$CLAP_BN0_CONSTANTS" "\$TEMP_DIR/work_dir/weights/clap_bn0_constants.npz"
cp -r "$ROBERTA_PATH/." "\$TEMP_DIR/roberta-base/"
[ -d "\$TARGET_GLOBAL" ] && cp -r "\$TARGET_GLOBAL/." "\$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$AUDIO_FORMAT/$target_folder/"
cp "$DATASEC_GLOBAL/RAW_DATASET/raw_$AUDIO_FORMAT"/*.h5 "\$TEMP_DIR/dataSEC/RAW_DATASET/raw_$AUDIO_FORMAT/" 2>/dev/null

# Stage-in selettivo del modello generativo
if [ "$GENERATIVE_BACKBONE" = "vae" ] || [ "$GENERATIVE_BACKBONE" = "vae_refiner" ]; then
    export VAE_TARGET_EPOCH=$VAE_TARGET_EPOCH
    VAE_ZERO_IDX=\$(( VAE_TARGET_EPOCH - 1 ))
    VAE_STAGED=""
    if [ -f "$VAE_MODELS_PATH/dual_vae_epoch_\${VAE_ZERO_IDX}.pt" ]; then
        VAE_STAGED="$VAE_MODELS_PATH/dual_vae_epoch_\${VAE_ZERO_IDX}.pt"
    elif [ -f "$VAE_MODELS_PATH/dual_vae_epoch_\${VAE_TARGET_EPOCH}.pt" ]; then
        VAE_STAGED="$VAE_MODELS_PATH/dual_vae_epoch_\${VAE_TARGET_EPOCH}.pt"
    else
        VAE_STAGED=\$(ls -v "$VAE_MODELS_PATH"/dual_vae_epoch_*.pt 2>/dev/null | tail -n 1)
    fi
    echo "📦 Staging Single VAE Checkpoint: \$VAE_STAGED"
    cp "\$VAE_STAGED" "\$TEMP_DIR/work_dir/vae_model/dual_vae_epoch_\${VAE_ZERO_IDX}.pt"

    if [ "$GENERATIVE_BACKBONE" = "vae_refiner" ]; then
        export REFINER_TARGET_EPOCH=$REFINER_TARGET_EPOCH
        REF_ZERO_IDX=\$(( REFINER_TARGET_EPOCH - 1 ))
        REF_STAGED=""
        if [ -f "$REFINER_MODELS_PATH/refiner_epoch_\${REF_ZERO_IDX}.pt" ]; then
            REF_STAGED="$REFINER_MODELS_PATH/refiner_epoch_\${REF_ZERO_IDX}.pt"
        elif [ -f "$REFINER_MODELS_PATH/refiner_epoch_\${REFINER_TARGET_EPOCH}.pt" ]; then
            REF_STAGED="$REFINER_MODELS_PATH/refiner_epoch_\${REFINER_TARGET_EPOCH}.pt"
        else
            REF_STAGED=\$(ls -v "$REFINER_MODELS_PATH"/refiner_epoch_*.pt 2>/dev/null | tail -n 1)
        fi
        echo "📦 Staging Single Refiner Checkpoint: \$REF_STAGED"
        cp "\$REF_STAGED" "\$TEMP_DIR/work_dir/refiner_model/refiner_epoch_\${REF_ZERO_IDX}.pt"
    fi
else
    export TARGET_EPOCH=500
    cp "$DIFFUSION_MODELS_PATH/unet_epoch_\$(( TARGET_EPOCH - 1 )).pt" "\$TEMP_DIR/work_dir/diff_model/" 2>/dev/null || \\
    cp "\$(ls -v "$DIFFUSION_MODELS_PATH"/unet_epoch_*.pt 2>/dev/null | tail -n 1)" "\$TEMP_DIR/work_dir/diff_model/" 2>/dev/null
fi

cat << 'INNER_EOF' > "\$TEMP_DIR/work_dir/join_logs_wrapper.py"
import sys, os
sys.path.append('/app')
from src.utils import join_logs
from src.dirs_config import basedir_preprocessed
def main():
    suffix = "${N_OCTAVE}_octave_no_inject" if int("${N_OCTAVE}") != 0 and "${INJECT_OCTAVE_CMD}" == "False" else "${N_OCTAVE}_octave"
    base_path = os.path.join(basedir_preprocessed, "$AUDIO_FORMAT", suffix)
    if not os.path.exists(base_path): return
    for entry in os.listdir(base_path):
        target_dir = os.path.join(base_path, entry)
        if os.path.isdir(target_dir) and entry.endswith("_secs"):
            join_logs(target_dir)
if __name__ == '__main__': main()
INNER_EOF

export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
export HF_HUB_OFFLINE=1
export CLAP_TEXT_ENCODER_PATH="/tmp_data/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
export LOCAL_CLAP_BN0_CONSTANTS_PATH="/tmp_data/work_dir/weights/clap_bn0_constants.npz"
export GENERATIVE_BACKBONE="$GENERATIVE_BACKBONE"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=\$(hostname)
export MASTER_PORT=\$(expr 20000 + \${SLURM_JOB_ID} % 10000)
export INJECT_OCTAVE="$INJECT_OCTAVE_CMD"

echo "🚀 Launching Embedding Extraction on 4 GPUs (Backbone: $GENERATIVE_BACKBONE)..."
srun --unbuffered -l -n 4 --export=ALL --cpu-bind=none \\
    singularity exec --nv --no-home --bind "/leonardo_scratch:/leonardo_scratch" --bind "\$TEMP_DIR:/tmp_data" \\
    --bind "\$(pwd):/app" --pwd "/app" "$SIF_FILE" \\
    python3 scripts/get_clap_embeddings.py --config_file "$CONFIG_FILE" --n_octave "$N_OCTAVE" --audio_format "$AUDIO_FORMAT"

finalize_and_cleanup
EOF
    chmod +x "$script"
    sbatch "$script"
}

run_slurm

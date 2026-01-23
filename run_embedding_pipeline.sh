#!/bin/bash
# run_pipeline.sh

# Manual pipeline orchestrator for Leonardo Cluster.
# Implements NVMe Local Staging to maximize I/O throughput.
# Replicates the production environment: offline mode, local weights, and isolated cache.

# Args:
#  - config_file (str): YAML configuration file;
#  - audio_format (str): Audio format (wav, mp3, flac);
#  - n_octave (int): Octave resolution;
#  - mode (str): 'interactive' or 'slurm'.

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <config_file> <audio_format> <n_octave> <mode>"
    echo "Modes: 'interactive' or 'slurm'"
    exit 1
fi

CONFIG_FILE=$1
AUDIO_FORMAT=$2
N_OCTAVE=$3
MODE=$4

# Global Production Assets
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
ROBERTA_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/roberta-base"

# Permanent global storage (Sibling of SEC_pipeline)
DATASEC_GLOBAL="/leonardo_scratch/large/userexternal/$USER/dataSEC"

run_interactive() {
    echo "🎬 Starting INTERACTIVE execution on local node with NVMe staging..."
    
    # 🎯 1. LOCAL STORAGE SETUP (Host-side)
    # Using local scratch if available, otherwise falling back to temporary global scratch
    LOCAL_JOB_DIR="/scratch_local/interactive_$$"
    if [ ! -d "/scratch_local" ]; then
        LOCAL_JOB_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_interactive_$$"
    fi
    
    mkdir -p "$LOCAL_JOB_DIR/dataSEC/RAW_DATASET"
    mkdir -p "$LOCAL_JOB_DIR/dataSEC/PREPROCESSED_DATASET"
    mkdir -p "$LOCAL_JOB_DIR/work_dir/roberta-base"
    mkdir -p "$LOCAL_JOB_DIR/work_dir/weights"
    mkdir -p "$LOCAL_JOB_DIR/numba_cache"

    # 🎯 2. STAGE-IN: Copy assets to local NVMe
    echo "📦 Staging-in data to local NVMe..."
    cp -r "$DATASEC_GLOBAL/RAW_DATASET/raw_$AUDIO_FORMAT" "$LOCAL_JOB_DIR/dataSEC/RAW_DATASET/"
    cp -r "$ROBERTA_PATH/." "$LOCAL_JOB_DIR/work_dir/roberta-base/"
    cp "$CLAP_SCRATCH_WEIGHTS" "$LOCAL_JOB_DIR/work_dir/weights/CLAP_weights_2023.pth"

    # 🎯 3. EXPORTS: Redirect Python using NODE_TEMP_BASE_DIR
    export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
    export HF_HUB_OFFLINE=1
    export CLAP_TEXT_ENCODER_PATH="/tmp_data/work_dir/roberta-base"
    export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
    export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
    export PYTHONUNBUFFERED=1
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    export MASTER_PORT=$(expr 29500 + $$ % 100)

    echo "🚀 Launching Singularity container on NVMe..."
    singularity exec --nv \
        --bind "$LOCAL_JOB_DIR:/tmp_data" \
        --bind "$(pwd):/app" \
        --pwd "/app" \
        "$SIF_FILE" \
        python3 scripts/get_clap_embeddings.py --config_file "$CONFIG_FILE" --n_octave "$N_OCTAVE" --audio_format "$AUDIO_FORMAT"

    echo "🔗 Joining HDF5 files on NVMe..."
    singularity exec --nv \
        --bind "$LOCAL_JOB_DIR:/tmp_data" \
        --bind "$(pwd):/app" \
        --pwd "/app" \
        "$SIF_FILE" \
        python3 scripts/join_hdf5.py --config_file "$CONFIG_FILE" --n_octave "$N_OCTAVE" --audio_format "$AUDIO_FORMAT"

    # 🎯 4. STAGE-OUT: Persistent save
    echo "📦 Staging-out results to global scratch..."
    mkdir -p "$DATASEC_GLOBAL/PREPROCESSED_DATASET/$AUDIO_FORMAT/${N_OCTAVE}_octave"
    cp -r "$LOCAL_JOB_DIR/dataSEC/PREPROCESSED_DATASET/$AUDIO_FORMAT/${N_OCTAVE}_octave/." "$DATASEC_GLOBAL/PREPROCESSED_DATASET/$AUDIO_FORMAT/${N_OCTAVE}_octave/"

    rm -rf "$LOCAL_JOB_DIR"
}

run_slurm() {
    echo "📤 Dispatching SLURM job with NVMe staging..."
    local j_name="prod_${AUDIO_FORMAT}_oct${N_OCTAVE}"
    local script="submit_${j_name}.sh"

    cat << EOF > "$script"
#!/bin/bash
#SBATCH --job-name=$j_name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:59
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_Pb-skite
#SBATCH --output=%x_%j.out

# 🎯 1. LOCAL PATH DEFINITION (valido per ogni task)
LOCAL_JOB_DIR="/scratch_local/job_\${SLURM_JOB_ID}"

# 🎯 2. PREPARATION: NODE AND STAGE-IN
echo "📦 Staging-in data to local NVMe on all tasks..."
srun --ntasks=\$SLURM_NTASKS bash -c "mkdir -p \$LOCAL_JOB_DIR/dataSEC/RAW_DATASET && \
    mkdir -p \$LOCAL_JOB_DIR/dataSEC/PREPROCESSED_DATASET && \
    mkdir -p \$LOCAL_JOB_DIR/work_dir/roberta-base && \
    mkdir -p \$LOCAL_JOB_DIR/work_dir/weights && \
    mkdir -p \$LOCAL_JOB_DIR/numba_cache && \
    cp -r $DATASEC_GLOBAL/RAW_DATASET/raw_$AUDIO_FORMAT \$LOCAL_JOB_DIR/dataSEC/RAW_DATASET/ && \
    cp -r $ROBERTA_PATH/. \$LOCAL_JOB_DIR/work_dir/roberta-base/ && \
    cp $CLAP_SCRATCH_WEIGHTS \$LOCAL_JOB_DIR/work_dir/weights/CLAP_weights_2023.pth"

# 🎯 3. EXPORTS
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
export HF_HUB_OFFLINE=1
export CLAP_TEXT_ENCODER_PATH="/tmp_data/work_dir/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export MASTER_PORT=\$(expr 20000 + \${SLURM_JOB_ID} % 10000)

echo "🚀 Starting Parallel Embedding Pipeline..."
srun --unbuffered -l -n 4 --export=ALL --cpu-bind=none \\
    singularity exec --nv \\
    --bind "\$LOCAL_JOB_DIR:/tmp_data" \\
    --bind "\$(pwd):/app" \\
    --pwd "/app" \\
    "$SIF_FILE" \\
    python3 scripts/get_clap_embeddings.py --config_file "$CONFIG_FILE" --n_octave "$N_OCTAVE" --audio_format "$AUDIO_FORMAT"

echo "🔗 Joining HDF5 files..."
singularity exec --nv \\
    --bind "\$LOCAL_JOB_DIR:/tmp_data" \\
    --bind "\$(pwd):/app" \\
    --pwd "/app" \\
    "$SIF_FILE" \\
    python3 scripts/join_hdf5.py --config_file "$CONFIG_FILE" --n_octave "$N_OCTAVE" --audio_format "$AUDIO_FORMAT"

# 🎯 4. STAGE-OUT
echo "📦 Staging-out results to global scratch..."
TARGET_GLOBAL="$DATASEC_GLOBAL/PREPROCESSED_DATASET/$AUDIO_FORMAT/${N_OCTAVE}_octave"
mkdir -p "\$TARGET_GLOBAL"
cp -r "\$LOCAL_JOB_DIR/dataSEC/PREPROCESSED_DATASET/$AUDIO_FORMAT/${N_OCTAVE}_octave/." "\$TARGET_GLOBAL/"

# Final cleanup
srun --ntasks=\$SLURM_NTASKS rm -rf "\$LOCAL_JOB_DIR"
EOF

    chmod +x "$script"
    sbatch "$script"
}

case $MODE in
    "interactive") run_interactive ;;
    "slurm") run_slurm ;;
    *) echo "❌ Error: Invalid mode." && exit 1 ;;
esac

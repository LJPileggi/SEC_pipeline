if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <config_file> <audio_format> <n_octave> <mode>"
    exit 1
fi

CONFIG_FILE=$1
AUDIO_FORMAT=$2
N_OCTAVE=$3
MODE=$4

SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
ROBERTA_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/roberta-base"
DATASEC_GLOBAL="/leonardo_scratch/large/userexternal/$USER/dataSEC"

run_slurm() {
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

# 🎯 1. PATH DEFINITION (Absolute scratch paths)
# We use a temporary directory on the node's local storage or high-speed scratch.
TEMP_DIR="/leonardo_scratch/large/userexternal/\$USER/tmp_job_\$SLURM_JOB_ID"
mkdir -p "\$TEMP_DIR/dataSEC/RAW_DATASET"
mkdir -p "\$TEMP_DIR/work_dir/weights"
mkdir -p "\$TEMP_DIR/roberta-base"
mkdir -p "\$TEMP_DIR/numba_cache"

# 🎯 2. STAGE-IN: Move data to the temporary workspace
echo "📦 Staging-in data..."
cp -r "$DATASEC_GLOBAL/RAW_DATASET/raw_$AUDIO_FORMAT" "\$TEMP_DIR/dataSEC/RAW_DATASET/"
cp -r "$ROBERTA_PATH/." "\$TEMP_DIR/roberta-base/"
cp "$CLAP_SCRATCH_WEIGHTS" "\$TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"

# 🎯 3. ENVIRONMENT REDIRECTION (The Mantra)
# We point everything to /tmp_data which is our container mount point.
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
export HF_HUB_OFFLINE=1
export CLAP_TEXT_ENCODER_PATH="/tmp_data/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export MASTER_PORT=\$(expr 20000 + \${SLURM_JOB_ID} % 10000)

# 🎯 4. EXECUTION
echo "🚀 Starting Parallel Embedding Pipeline..."
# We bind the pre-existing TEMP_DIR to /tmp_data inside the container
srun --unbuffered -l -n 4 --export=ALL --cpu-bind=none \\
    singularity exec --nv \\
    --bind "/leonardo_scratch:/leonardo_scratch" \\
    --bind "\$TEMP_DIR:/tmp_data" \\
    --bind "\$(pwd):/app" \\
    --pwd "/app" \\
    "$SIF_FILE" \\
    python3 scripts/get_clap_embeddings.py --config_file "$CONFIG_FILE" --n_octave "$N_OCTAVE" --audio_format "$AUDIO_FORMAT"

echo "🔗 Joining HDF5 files..."
singularity exec --nv \\
    --bind "/leonardo_scratch:/leonardo_scratch" \\
    --bind "\$TEMP_DIR:/tmp_data" \\
    --bind "\$(pwd):/app" \\
    --pwd "/app" \\
    "$SIF_FILE" \\
    python3 scripts/join_hdf5.py --config_file "$CONFIG_FILE" --n_octave "$N_OCTAVE" --audio_format "$AUDIO_FORMAT"

# 🎯 5. STAGE-OUT
echo "📦 Staging-out results..."
TARGET_GLOBAL="$DATASEC_GLOBAL/PREPROCESSED_DATASET/$AUDIO_FORMAT/${N_OCTAVE}_octave"
mkdir -p "\$TARGET_GLOBAL"
cp -r "\$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$AUDIO_FORMAT/${N_OCTAVE}_octave/." "\$TARGET_GLOBAL/"

# Cleanup
rm -rf "\$TEMP_DIR"
EOF

    chmod +x "$script"
    sbatch "$script"
}

case $MODE in
    "interactive") echo "Interactive mode not updated for NVMe yet. Use slurm." ;;
    "slurm") run_slurm ;;
esac

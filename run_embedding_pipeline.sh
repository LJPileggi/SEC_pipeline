#!/bin/bash
# run_pipeline.sh

# Manual pipeline orchestrator for Leonardo.
# Supports both interactive execution on allocated nodes and single Slurm job submission.
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
# Define the physical path to the dataset on the host
REAL_DATA_BASE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/dataSEC"

run_interactive() {
    echo "🎬 Starting INTERACTIVE execution on local node..."
    
    # Setup temporary directory based on PID to avoid resource contention
    TEMP_BASE="/leonardo_scratch/large/userexternal/$USER/tmp_interactive_$$"
    mkdir -p "$TEMP_BASE/work_dir/huggingface/hub/models--microsoft--msclap/snapshots/main"
    mkdir -p "$TEMP_BASE/roberta-base"
    # Ensure the mount source exists
    mkdir -p "$REAL_DATA_BASE"

    # Preparing local weights for the container
    cp "$CLAP_SCRATCH_WEIGHTS" "$TEMP_BASE/work_dir/huggingface/hub/models--microsoft--msclap/snapshots/main/CLAP_weights_2023.pth"
    cp -r "$ROBERTA_PATH/." "$TEMP_BASE/roberta-base/"

    # Environment variables for firewall-safe processing
    export HF_HOME="$TEMP_BASE/work_dir/huggingface"
    export HF_HUB_OFFLINE=1
    export CLAP_TEXT_ENCODER_PATH="/tmp_data/roberta-base"
    export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/huggingface/hub/models--microsoft--msclap/snapshots/main/CLAP_weights_2023.pth"
    export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"

    echo "🚀 Launching Singularity container..."
    singularity exec --nv \
        --bind "/leonardo_scratch:/leonardo_scratch" \
        --bind "$TEMP_BASE:/tmp_data" \
        --bind "$REAL_DATA_BASE:/tmp_data/dataSEC" \
        --bind "$(pwd):/app" \
        --pwd "/app" \
        "$SIF_FILE" \
        python3 scripts/get_clap_embeddings.py \
            --config_file "$CONFIG_FILE" \
            --n_octave "$N_OCTAVE" \
            --audio_format "$AUDIO_FORMAT"

    echo "Joining HDF5 files..."
    singularity exec --nv \
        --bind "/leonardo_scratch:/leonardo_scratch" \
        --bind "$TEMP_BASE:/tmp_data" \
        --bind "$REAL_DATA_BASE:/tmp_data/dataSEC" \
        --bind "$(pwd):/app" \
        --pwd "/app" \
        "$SIF_FILE" \
        python3 scripts/join_hdf5.py \
            --config_file "$CONFIG_FILE" \
            --n_octave "$N_OCTAVE" \
            --audio_format "$AUDIO_FORMAT"

    # Workspace cleanup
    echo "🧹 Cleaning up temporary data..."
    rm -rf "$TEMP_BASE"
}

run_slurm() {
    echo "📤 Dispatching single Slurm job..."
    local j_name="manual_${AUDIO_FORMAT}_oct${N_OCTAVE}"
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

JOB_WORK_DIR="/leonardo_scratch/large/userexternal/\$USER/tmp_job_\${SLURM_JOB_ID}"
mkdir -p "\$JOB_WORK_DIR/work_dir/huggingface/hub/models--microsoft--msclap/snapshots/main"
mkdir -p "\$JOB_WORK_DIR/roberta-base"

cp "$CLAP_SCRATCH_WEIGHTS" "\$JOB_WORK_DIR/work_dir/huggingface/hub/models--microsoft--msclap/snapshots/main/CLAP_weights_2023.pth"
cp -r "$ROBERTA_PATH/." "\$JOB_WORK_DIR/roberta-base/"

export HF_HOME="\$JOB_WORK_DIR/work_dir/huggingface"
export HF_HUB_OFFLINE=1
export CLAP_TEXT_ENCODER_PATH="/tmp_data/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/huggingface/hub/models--microsoft--msclap/snapshots/main/CLAP_weights_2023.pth"
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"

srun --unbuffered -l -n 4 --export=ALL --cpu-bind=none \\
    singularity exec --nv \\
    --bind "/leonardo_scratch:/leonardo_scratch" \\
    --bind "\$JOB_WORK_DIR:/tmp_data" \\
    --bind "$REAL_DATA_BASE:/tmp_data/dataSEC" \\
    --bind "\$(pwd):/app" \\
    --pwd "/app" \\
    "$SIF_FILE" \\
    python3 scripts/get_clap_embeddings.py --config_file "$CONFIG_FILE" --n_octave "$N_OCTAVE" --audio_format "$AUDIO_FORMAT"

echo "Joining HDF5 files..."
singularity exec --nv \\
    --bind "/leonardo_scratch:/leonardo_scratch" \\
    --bind "\$JOB_WORK_DIR:/tmp_data" \\
    --bind "$REAL_DATA_BASE:/tmp_data/dataSEC" \\
    --bind "\$(pwd):/app" \\
    --pwd "/app" \\
    "$SIF_FILE" \\
    python3 scripts/join_hdf5.py --config_file "$CONFIG_FILE" --n_octave "$N_OCTAVE" --audio_format "$AUDIO_FORMAT"

rm -rf "\$JOB_WORK_DIR"
EOF

    chmod +x "$script"
    sbatch "$script"
    echo "✅ Job submitted successfully."
}

# Execution Dispatcher
case $MODE in
    "interactive") run_interactive ;;
    "slurm") run_slurm ;;
    *) echo "❌ Error: Invalid mode. Use 'interactive' or 'slurm'." && exit 1 ;;
esac

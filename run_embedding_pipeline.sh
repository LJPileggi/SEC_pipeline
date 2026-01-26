#!/bin/bash
# run_embedding_pipeline.sh

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <config_file> <audio_format> <n_octave> <mode> [world_size]"
    echo "Modes: slurm, local"
    exit 1
fi

CONFIG_FILE=$1
AUDIO_FORMAT=$2
N_OCTAVE=$3
MODE=$4
WORLD_SIZE=${5:-4} # Default to 4 processes for local mode

SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
ROBERTA_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/roberta-base"
DATASEC_GLOBAL="/leonardo_scratch/large/userexternal/$USER/dataSEC"

# 🎯 SHARED POST-PROCESSING LOGIC
# This helper creates the Python wrapper used for log merging
create_log_wrapper() {
    local target_path=$1
    cat << 'INNER_EOF' > "$target_path"
import sys, os
sys.path.append('/app')
from src.utils import join_logs
from src.dirs_config import basedir_preprocessed
def main():
    base_path = os.path.join(basedir_preprocessed, "$AUDIO_FORMAT", "${N_OCTAVE}_octave")
    if not os.path.exists(base_path): return
    for entry in os.listdir(base_path):
        target_dir = os.path.join(base_path, entry)
        if os.path.isdir(target_dir) and entry.endswith("_secs"):
            join_logs(target_dir)
if __name__ == '__main__': main()
INNER_EOF
}

# ------------------------------------------------------------------------------
# SLURM MODE
# ------------------------------------------------------------------------------
run_slurm() {
    local j_name="prod_${AUDIO_FORMAT}_oct${N_OCTAVE}"
    local script="submit_${j_name}.sh"
    local FINAL_DEST="$DATASEC_GLOBAL/PREPROCESSED_DATASET/$AUDIO_FORMAT/${N_OCTAVE}_octave"

    cat << EOF > "$script"
#!/bin/bash
#SBATCH --job-name=$j_name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_Pb-skite
#SBATCH --output=%x_%j.out

TEMP_DIR="/leonardo_scratch/large/userexternal/\$USER/tmp_job_\$SLURM_JOB_ID"
TARGET_GLOBAL="$FINAL_DEST"

mkdir -p "\$TEMP_DIR/dataSEC/RAW_DATASET/raw_$AUDIO_FORMAT"
mkdir -p "\$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$AUDIO_FORMAT/${N_OCTAVE}_octave"
mkdir -p "\$TEMP_DIR/work_dir/weights"
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
        rsync -rlt "\$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$AUDIO_FORMAT/${N_OCTAVE}_octave/" "\$TARGET_GLOBAL/"
        rm -rf "\$TEMP_DIR"
    fi
    exit 0
}
trap 'finalize_and_cleanup' SIGTERM SIGINT

echo "📦 Stage-in..."
cp "$CLAP_SCRATCH_WEIGHTS" "\$TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"
cp -r "$ROBERTA_PATH/." "\$TEMP_DIR/roberta-base/"
[ -d "\$TARGET_GLOBAL" ] && cp -r "\$TARGET_GLOBAL/." "\$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$AUDIO_FORMAT/${N_OCTAVE}_octave/"
cp "$DATASEC_GLOBAL/RAW_DATASET/raw_$AUDIO_FORMAT"/*.h5 "\$TEMP_DIR/dataSEC/RAW_DATASET/raw_$AUDIO_FORMAT/" 2>/dev/null

# Create wrapper inside the heredoc
cat << 'INNER_EOF' > "\$TEMP_DIR/work_dir/join_logs_wrapper.py"
import sys, os
sys.path.append('/app')
from src.utils import join_logs
from src.dirs_config import basedir_preprocessed
def main():
    base_path = os.path.join(basedir_preprocessed, "$AUDIO_FORMAT", "${N_OCTAVE}_octave")
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
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=\$(hostname)
export MASTER_PORT=\$(expr 20000 + \${SLURM_JOB_ID} % 10000)

srun --unbuffered -l -n 4 --export=ALL --cpu-bind=none \\
    singularity exec --nv --no-home --bind "/leonardo_scratch:/leonardo_scratch" --bind "\$TEMP_DIR:/tmp_data" \\
    --bind "\$(pwd):/app" --pwd "/app" "$SIF_FILE" \\
    python3 scripts/get_clap_embeddings.py --config_file "$CONFIG_FILE" --n_octave "$N_OCTAVE" --audio_format "$AUDIO_FORMAT"

finalize_and_cleanup
EOF
    chmod +x "$script"
    sbatch "$script"
}

# ------------------------------------------------------------------------------
# LOCAL MODE
# ------------------------------------------------------------------------------
run_local() {
    echo "🖥️  Starting Local Interactive Mode..."
    
    # Use a local temporary directory
    TEMP_DIR="/tmp/job_local_$(date +%s)"
    TARGET_GLOBAL="$DATASEC_GLOBAL/PREPROCESSED_DATASET/$AUDIO_FORMAT/${N_OCTAVE}_octave"

    mkdir -p "$TEMP_DIR/dataSEC/RAW_DATASET/raw_$AUDIO_FORMAT"
    mkdir -p "$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$AUDIO_FORMAT/${N_OCTAVE}_octave"
    mkdir -p "$TEMP_DIR/work_dir/weights"
    mkdir -p "$TEMP_DIR/roberta-base"
    mkdir -p "$TEMP_DIR/numba_cache"

    finalize_local() {
        trap - SIGINT
        echo -e "\n🛑 Local Interrupt! Consolidating..."
        # 1. Merge Logs
        singularity exec --no-home --bind "$TEMP_DIR:/tmp_data" --bind "$(pwd):/app" --pwd "/app" "$SIF_FILE" \
            python3 "/tmp_data/work_dir/join_logs_wrapper.py"
        # 2. Join HDF5
        singularity exec --nv --no-home --bind "/leonardo_scratch:/leonardo_scratch" --bind "$TEMP_DIR:/tmp_data" \
            --bind "$(pwd):/app" --pwd "/app" "$SIF_FILE" \
            python3 scripts/join_hdf5.py --config_file "$CONFIG_FILE" --n_octave "$N_OCTAVE" --audio_format "$AUDIO_FORMAT"
        # 3. Stage-out
        mkdir -p "$TARGET_GLOBAL"
        rsync -rlt "$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$AUDIO_FORMAT/${N_OCTAVE}_octave/" "$TARGET_GLOBAL/"
        rm -rf "$TEMP_DIR"
        echo "✅ Local consolidation done. Bye!"
        exit 0
    }
    trap 'finalize_local' SIGINT

    echo "📦 Staging-in data..."
    cp "$CLAP_SCRATCH_WEIGHTS" "$TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"
    cp -r "$ROBERTA_PATH/." "$TEMP_DIR/roberta-base/"
    [ -d "$TARGET_GLOBAL" ] && cp -r "$TARGET_GLOBAL/." "$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$AUDIO_FORMAT/${N_OCTAVE}_octave/"
    cp "$DATASEC_GLOBAL/RAW_DATASET/raw_$AUDIO_FORMAT"/*.h5 "$TEMP_DIR/dataSEC/RAW_DATASET/raw_$AUDIO_FORMAT/" 2>/dev/null
    
    create_log_wrapper "$TEMP_DIR/work_dir/join_logs_wrapper.py"

    export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
    export HF_HUB_OFFLINE=1
    export CLAP_TEXT_ENCODER_PATH="/tmp_data/roberta-base"
    export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
    export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
    export PYTHONUNBUFFERED=1
    export PYTORCH_ALLOC_CONF=expandable_segments:True

    echo "🚀 Running local worker..."
    singularity exec --nv --no-home --bind "/leonardo_scratch:/leonardo_scratch" --bind "$TEMP_DIR:/tmp_data" \
        --bind "$(pwd):/app" --pwd "/app" "$SIF_FILE" \
        python3 scripts/get_clap_embeddings.py --config_file "$CONFIG_FILE" --n_octave "$N_OCTAVE" --audio_format "$AUDIO_FORMAT" \
        --world_size "$WORLD_SIZE"

    finalize_local
}

# ------------------------------------------------------------------------------
# MAIN DISPATCHER
# ------------------------------------------------------------------------------
case $MODE in
    "slurm") run_slurm ;;
    "local") run_local ;;
    *) echo "Invalid mode. Use 'slurm' or 'local'." && exit 1 ;;
esac

#!/bin/bash
# run_embedding_pipeline.sh

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
    
    # 🎯 FIX: Risolviamo il percorso globale ORA per scriverlo fisicamente nel submit
    local FINAL_DEST="$DATASEC_GLOBAL/PREPROCESSED_DATASET/$AUDIO_FORMAT/${N_OCTAVE}_octave"

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

# 🎯 1. PATH DEFINITION
TEMP_DIR="/leonardo_scratch/large/userexternal/\$USER/tmp_job_\$SLURM_JOB_ID"
TARGET_GLOBAL="$FINAL_DEST"

mkdir -p "\$TEMP_DIR/dataSEC/RAW_DATASET/raw_$AUDIO_FORMAT"
mkdir -p "\$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$AUDIO_FORMAT/${N_OCTAVE}_octave"
mkdir -p "\$TEMP_DIR/work_dir/weights"
mkdir -p "\$TEMP_DIR/roberta-base"

# 🎯 2. SIGNAL TRAP (Emergency Stage-out only)
cleanup_and_stageout() {
    echo "⚠️ Signal caught! Starting emergency Stage-out..."
    if [ -d "\$TEMP_DIR" ]; then
        mkdir -p "\$TARGET_GLOBAL"
        # We use -rlt (recursive, links, times) to avoid conflicts linked to permissions/groups
        rsync -rlt "\$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$AUDIO_FORMAT/${N_OCTAVE}_octave/" "\$TARGET_GLOBAL/"
        rm -rf "\$TEMP_DIR"
    fi
    echo "✅ Emergency Stage-out completed."
    exit 0
}
trap 'cleanup_and_stageout' SIGTERM SIGINT

# 🎯 3. STAGE-IN (Data Locality)
echo "📦 Staging-in data and weights..."
cp "$CLAP_SCRATCH_WEIGHTS" "\$TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"
cp -r "$ROBERTA_PATH/." "\$TEMP_DIR/roberta-base/"

if [ -d "\$TARGET_GLOBAL" ]; then
    cp -r "\$TARGET_GLOBAL/." "\$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$AUDIO_FORMAT/${N_OCTAVE}_octave/"
fi

cp "$DATASEC_GLOBAL/RAW_DATASET/raw_$AUDIO_FORMAT"/*.h5 "\$TEMP_DIR/dataSEC/RAW_DATASET/raw_$AUDIO_FORMAT/" 2>/dev/null || \\
cp "$DATASEC_GLOBAL/RAW_DATASET/raw_$AUDIO_FORMAT"/*/*.h5 "\$TEMP_DIR/dataSEC/RAW_DATASET/raw_$AUDIO_FORMAT/"

# 🎯 4. ENVIRONMENT REDIRECTION (The Mantra)
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
export CLAP_TEXT_ENCODER_PATH="/tmp_data/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=\$(hostname)
export MASTER_PORT=\$(expr 20000 + \${SLURM_JOB_ID} % 10000)

# 🎯 5. EXECUTION
echo "🚀 Starting Parallel Embedding Pipeline..."
srun --unbuffered -l -n 4 --export=ALL --cpu-bind=none \\
    singularity exec --nv \\
    --bind "/leonardo_scratch:/leonardo_scratch" \\
    --bind "\$TEMP_DIR:/tmp_data" \\
    --bind "\$(pwd):/app" \\
    --pwd "/app" \\
    "$SIF_FILE" \\
    python3 scripts/get_clap_embeddings.py --config_file "$CONFIG_FILE" --n_octave "$N_OCTAVE" --audio_format "$AUDIO_FORMAT"

# 🎯 6. LOG MERGING
echo "🔗 Merging rank-specific logs..."
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

singularity exec --bind "\$TEMP_DIR:/tmp_data" --bind "\$(pwd):/app" --pwd "/app" "$SIF_FILE" \\
    python3 "/tmp_data/work_dir/join_logs_wrapper.py"

# 🎯 7. FINAL JOINING
echo "🔗 Joining HDF5 files..."
singularity exec --nv \\
    --bind "/leonardo_scratch:/leonardo_scratch" \\
    --bind "\$TEMP_DIR:/tmp_data" \\
    --bind "\$(pwd):/app" \\
    --pwd "/app" \\
    "$SIF_FILE" \\
    python3 scripts/join_hdf5.py --config_file "$CONFIG_FILE" --n_octave "$N_OCTAVE" --audio_format "$AUDIO_FORMAT"

# 🎯 8. FINAL STAGE-OUT (Normal Completion)
echo "🏁 Consolidating final results..."
mkdir -p "\$TARGET_GLOBAL"
rsync -rlt "\$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$AUDIO_FORMAT/${N_OCTAVE}_octave/" "\$TARGET_GLOBAL/"
rm -rf "\$TEMP_DIR"
echo "✅ Job Completed Successfully."
EOF

    chmod +x "$script"
    sbatch "$script"
}

case $MODE in
    "slurm") run_slurm ;;
    *) echo "Usage: $0 <config> <format> <oct> slurm" && exit 1 ;;
esac

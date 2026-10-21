#!/bin/bash
# ==========================================================
# RIGOROUS SEQUENTIAL SCHEDULER FOR LEONARDO CLUSTER
# Mantra: "Strict Parsing, Absolute Environment Parity, & Hard Staging"
# ==========================================================

DIRECTIVES_FILE=$1
CURRENT_USER=$(whoami)

if [ ! -f "$DIRECTIVES_FILE" ]; then
    echo "❌ Error: Directives file '$DIRECTIVES_FILE' not found."
    exit 1
fi

echo "📖 Parsing Global Header..."

parse_path() {
    local raw_val=$(grep "$1" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
    echo "$raw_val" | sed "s/large\/userexternal\/USER/large\/userexternal\/$CURRENT_USER/g"
}

SIF_FILE=$(parse_path "SIF_FILE")
CLAP_SCRATCH_WEIGHTS=$(parse_path "CLAP_WEIGHTS")
ROBERTA_PATH=$(parse_path "ROBERTA_PATH")
DATASEC_GLOBAL=$(parse_path "DATASEC_GLOBAL")

DIFF_MODELS_PATH=$(parse_path "DIFF_MODELS")
VAE_MODELS_PATH=$(parse_path "VAE_MODELS")
REFINER_MODELS_PATH=$(parse_path "REFINER_MODELS")

PARSED_BACKBONE=$(grep "GENERATIVE_BACKBONE" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
GENERATIVE_BACKBONE=${PARSED_BACKBONE:-vae_refiner}

SLURM_ACCOUNT=$(grep "SLURM_ACCOUNT" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
SLURM_PARTITION=$(grep "SLURM_PARTITION" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
SLURM_TIME=$(grep "SLURM_TIME" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)

submit_and_wait_task() {
    local c_file=$1; local fmt=$2; local oct=$3; local inj=$4
    local j_name="emb_${fmt}_oct${oct}_${GENERATIVE_BACKBONE}"
    local script="submit_${j_name}.sh"

    local target_folder="${oct}_octave"
    if [ "$oct" -ne 0 ] && [ "$inj" = "False" ]; then
        target_folder="${oct}_octave_no_inject"
    fi
    local FINAL_DEST="$DATASEC_GLOBAL/PREPROCESSED_DATASET/$fmt/$target_folder"

    cat << EOF > "$script"
#!/bin/bash
#SBATCH --job-name=$j_name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=$SLURM_TIME
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH -p $SLURM_PARTITION
#SBATCH -A $SLURM_ACCOUNT
#SBATCH --output=${j_name}_%j.out
#SBATCH --signal=B:TERM@60

export TEMP_DIR="/leonardo_scratch/large/userexternal/$CURRENT_USER/tmp_job_\$SLURM_JOB_ID"
TARGET_GLOBAL="$FINAL_DEST"

mkdir -p "\$TEMP_DIR/dataSEC/RAW_DATASET/raw_$fmt"
mkdir -p "\$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$fmt/$target_folder"
mkdir -p "\$TEMP_DIR/work_dir/weights"
mkdir -p "\$TEMP_DIR/work_dir/diff_model"
mkdir -p "\$TEMP_DIR/work_dir/vae_model"
mkdir -p "\$TEMP_DIR/work_dir/refiner_model"
mkdir -p "\$TEMP_DIR/roberta-base"
mkdir -p "\$TEMP_DIR/numba_cache"

CLEANED_UP=0

finalize_and_cleanup() {
    local sig=\$1
    if [ "\$CLEANED_UP" -eq 1 ]; then
        return
    fi
    CLEANED_UP=1

    trap - SIGTERM SIGINT SIGHUP EXIT
    echo -e "\n⚠️ Signal [\${sig:-EXIT}] intercettato! Avvio consolidamento e salvataggio..."

    if [ -n "\$SRUN_PID" ] && kill -0 "\$SRUN_PID" 2>/dev/null; then
        echo "🛑 Arresto controllato del processo srun (PID: \$SRUN_PID)..."
        kill -SIGTERM "\$SRUN_PID" 2>/dev/null
        sleep 3
    fi

    if [ -d "\$TEMP_DIR" ]; then
        echo "🔗 Merging logs & Joining HDF5..."
        singularity exec --no-home --bind "\$TEMP_DIR:/tmp_data" --bind "\$(pwd):/app" --pwd "/app" "$SIF_FILE" \\
            python3 "/tmp_data/work_dir/join_logs_wrapper.py"
        singularity exec --nv --no-home --bind "/leonardo_scratch:/leonardo_scratch" --bind "\$TEMP_DIR:/tmp_data" \\
            --bind "\$(pwd):/app" --pwd "/app" "$SIF_FILE" \\
            python3 scripts/join_hdf5.py --config_file "$c_file" --n_octave "$oct" --audio_format "$fmt"
        
        echo "📦 Stage-out: Syncing embeddings into Lustre storage..."
        mkdir -p "\$TARGET_GLOBAL"
        rsync -rlt --ignore-errors "\$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$fmt/$target_folder/" "\$TARGET_GLOBAL/"
        sync
        echo "🧹 Purging temporary scratch directory..."
        rm -rf "\$TEMP_DIR"
    fi
    echo "🏁 Consolidamento task completato con successo."
    exit 0
}

trap 'finalize_and_cleanup SIGTERM' SIGTERM
trap 'finalize_and_cleanup SIGINT'  SIGINT
trap 'finalize_and_cleanup SIGHUP'  SIGHUP
trap 'finalize_and_cleanup EXIT'    EXIT

echo "📦 Staging data..."
cp "$CLAP_SCRATCH_WEIGHTS" "\$TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth" || exit 1
cp -r "$ROBERTA_PATH/." "\$TEMP_DIR/roberta-base/" || exit 1

# Stage-in condizionato dei modelli generativi
if [ "$GENERATIVE_BACKBONE" = "vae" ] || [ "$GENERATIVE_BACKBONE" = "vae_refiner" ]; then
    export VAE_TARGET_EPOCH=50
    VAE_ZERO_IDX=\$(( VAE_TARGET_EPOCH - 1 ))
    LATEST_VAE=\$(ls -v "$VAE_MODELS_PATH"/dual_vae_epoch_*.pt 2>/dev/null | tail -n 1)
    [ -f "\$LATEST_VAE" ] && cp "\$LATEST_VAE" "\$TEMP_DIR/work_dir/vae_model/dual_vae_epoch_\${VAE_ZERO_IDX}.pt" || exit 1

    if [ "$GENERATIVE_BACKBONE" = "vae_refiner" ]; then
        export REFINER_TARGET_EPOCH=25
        REF_ZERO_IDX=\$(( REFINER_TARGET_EPOCH - 1 ))
        LATEST_REF=\$(ls -v "$REFINER_MODELS_PATH"/refiner_epoch_*.pt 2>/dev/null | tail -n 1)
        [ -f "\$LATEST_REF" ] && cp "\$LATEST_REF" "\$TEMP_DIR/work_dir/refiner_model/refiner_epoch_\${REF_ZERO_IDX}.pt" || exit 1
    fi
else
    export TARGET_EPOCH=500
    if [ -d "$DIFF_MODELS_PATH" ]; then
        echo "📦 Staging diffusion model weights from $DIFF_MODELS_PATH..."
        cp "$DIFF_MODELS_PATH/unet_epoch_\$(( TARGET_EPOCH - 1 )).pt" "\$TEMP_DIR/work_dir/diff_model/" 2>/dev/null || \\
        cp "\$(ls -v "$DIFF_MODELS_PATH"/unet_epoch_*.pt 2>/dev/null | tail -n 1)" "\$TEMP_DIR/work_dir/diff_model/" || exit 1
    else
        echo "⚠️ Warning: Diffusion models path '$DIFF_MODELS_PATH' not found!"
    fi
fi

if [ -d "\$TARGET_GLOBAL" ]; then
    echo "🔄 Resuming: copying existing embeddings..."
    cp -r "\$TARGET_GLOBAL/." "\$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$fmt/$target_folder/"
fi

RAW_SOURCE="$DATASEC_GLOBAL/RAW_DATASET/raw_$fmt"
if [ -d "\$RAW_SOURCE" ]; then
    cp "\$RAW_SOURCE"/*.h5 "\$TEMP_DIR/dataSEC/RAW_DATASET/raw_$fmt/"
else
    echo "❌ ERROR: Raw source \$RAW_SOURCE not found!"
    exit 1
fi

cat << 'INNER_EOF' > "\$TEMP_DIR/work_dir/join_logs_wrapper.py"
import sys, os
sys.path.append('/app')
from src.utils import join_logs
from src.dirs_config import basedir_preprocessed
def main():
    suffix = "${oct}_octave_no_inject" if int("${oct}") != 0 and "${inj}" == "False" else "${oct}_octave"
    base_path = os.path.join(basedir_preprocessed, "$fmt", suffix)
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
export DIFF_MODELS="/tmp_data/work_dir/diff_model"
export GENERATIVE_BACKBONE="$GENERATIVE_BACKBONE"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=\$(hostname)
export MASTER_PORT=\$(expr 20000 + \${SLURM_JOB_ID} % 10000)
export INJECT_OCTAVE="${inj}"

echo "🚀 Starting Parallel Embedding Pipeline ($GENERATIVE_BACKBONE)..."
srun --unbuffered -l -n 4 --export=ALL --cpu-bind=none \\
    singularity exec --nv --no-home --bind "/leonardo_scratch:/leonardo_scratch" --bind "\$TEMP_DIR:/tmp_data" \\
    --bind "\$(pwd):/app" --pwd "/app" "$SIF_FILE" \\
    python3 scripts/get_clap_embeddings.py --config_file "$c_file" --n_octave "$oct" --audio_format "$fmt" &

SRUN_PID=\$!
wait "\$SRUN_PID"
SRUN_STATUS=\$?

exit \$SRUN_STATUS
EOF

    chmod +x "$script"
    echo "⏳ Submitting $j_name and waiting for completion..."
    sbatch --wait "$script"
}

echo "🚀 Dispatching tasks SEQUENTIALLY..."
sed -n '/^[^#]/p' "$DIRECTIVES_FILE" | grep "|" | while IFS='|' read -r raw_cfg raw_fmt raw_oct raw_inj; do
    cfg=$(echo "$raw_cfg" | xargs)
    fmt=$(echo "$raw_fmt" | xargs)
    oct=$(echo "$raw_oct" | xargs)
    inj=$(echo "$raw_inj" | xargs)

    if [ -z "$inj" ]; then
        inj="False"
    fi

    [[ "$cfg" == *"SIF_FILE"* || "$cfg" == *"CLAP_WEIGHTS"* || "$cfg" == *"ROBERTA_PATH"* || "$cfg" == *"SLURM_"* || "$cfg" == *"SCHEDULING_MODE"* || "$cfg" == *"DATASEC_GLOBAL"* || "$cfg" == *"DIFF_MODELS"* || "$cfg" == *"VAE_MODELS"* || "$cfg" == *"REFINER_MODELS"* || "$cfg" == *"GENERATIVE_BACKBONE"* ]] && continue
    [ -z "$cfg" ] || [ -z "$fmt" ] || [ -z "$oct" ] && continue

    echo "➡️ Processing: $cfg | Format: $fmt | Octave: $oct | Inject Octave: $inj | Backbone: $GENERATIVE_BACKBONE"
    submit_and_wait_task "$cfg" "$fmt" "$oct" "$inj"
    echo "✅ Task finished."
done

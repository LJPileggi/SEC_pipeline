export PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
export SIF_FILE="$PROJECT_DIR/.containers/clap_pipeline.sif"

# Trova il percorso assoluto reale del file di conversione
SCRIPT_PATH="$(find "$PROJECT_DIR" -name "convert_ESC50_to_hdf5.py" 2>/dev/null | head -n 1)"
[ -z "$SCRIPT_PATH" ] && SCRIPT_PATH="$(pwd)/convert_ESC50_to_hdf5.py"

singularity exec --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$PROJECT_DIR:/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 "$SCRIPT_PATH"

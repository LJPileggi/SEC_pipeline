export PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
export SIF_FILE="$PROJECT_DIR/.containers/clap_pipeline.sif"
TARGET_SCRIPT="$(find /leonardo_scratch/large/userexternal/$USER -name "convert_ESC50_to_hdf5.py" 2>/dev/null | head -n 1)"

# Cartella di cache scrivibile per numba
mkdir -p "/leonardo_scratch/large/userexternal/$USER/tmp_numba_cache"
export NUMBA_CACHE_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_numba_cache"

singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --env NUMBA_CACHE_DIR="$NUMBA_CACHE_DIR" \
    "$SIF_FILE" \
    python3 "$TARGET_SCRIPT"

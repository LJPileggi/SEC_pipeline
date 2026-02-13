#!/bin/bash

# --- 1. CONFIGURATION ---
PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
LUSTRE_TMP="${PROJECT_DIR}/.tmp_io_bench"
STREAM_LOG="${LUSTRE_TMP}/io_results.log"
SIF_FILE="${PROJECT_DIR}/.containers/clap_pipeline.sif"

export N_FILES=500
export SR=44100
export DUR=1

# Pulizia totale
rm -rf "$LUSTRE_TMP"
mkdir -p "${LUSTRE_TMP}/wav_files"
touch "$STREAM_LOG"

# --- 2. GENERATION PHASE ---
echo "🔨 Phase 0: Generating data..."
singularity exec --no-home --bind "${LUSTRE_TMP}:/mnt_lustre" "$SIF_FILE" \
    python3 -u - <<'PY'
import numpy as np
import soundfile as sf
import h5py
import os
n, sr, d = int(os.environ['N_FILES']), int(os.environ['SR']), int(os.environ['DUR'])
wav_dir, h5_p = "/mnt_lustre/wav_files", "/mnt_lustre/dataset.h5"
data = np.random.uniform(-1, 1, sr * d).astype(np.float32)
with h5py.File(h5_p, 'w') as h5:
    ds = h5.create_dataset('audio', (n, sr*d), dtype='f4')
    for i in range(n):
        sf.write(f"{wav_dir}/track_{i}.wav", data, sr)
        ds[i] = data
    h5.flush()
print("✅ Python Generation Done.")
PY

# --- 3. GENERATE SLURM BATCH SCRIPT ---
# Usiamo 'EOF' per non far espandere nulla al login node
cat << 'EOF' > "${LUSTRE_TMP}/io_test_slurm.sh"
#!/bin/bash
#SBATCH --job-name=io_bench
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH -A IscrC_Pb-skite
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Recuperiamo le variabili passate o settiamole fisse
L_TMP="/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.tmp_io_bench"
LOG="/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.tmp_io_bench/io_results.log"
SIF="/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.containers/clap_pipeline.sif"
N=500

echo "🚀 NODE START: $(date)" >> "$LOG"

# --- Python Probe (Salvato su Lustre) ---
cat << 'PY_INNER' > "/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.tmp_io_bench/reader_probe.py"
import time, h5py, soundfile as sf, sys, os
wav_p, h5_p, label, n_files = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
print(f"\n--- I/O Test: {label} ---", flush=True)
try:
    s = time.perf_counter()
    for i in range(n_files):
        p = os.path.join(wav_p, f"track_{i}.wav")
        with sf.SoundFile(p) as f: _ = f.read()
    print(f"🔹 POSIX: {time.perf_counter() - s:.4f}s", flush=True)
    
    s = time.perf_counter()
    with h5py.File(h5_p, 'r') as h5:
        ds = h5['audio']
        for i in range(n_files): _ = ds[i]
    print(f"🔹 HDF5:  {time.perf_counter() - s:.4f}s", flush=True)
except Exception as e:
    print(f"❌ ERROR: {e}", flush=True)
PY_INNER

# Fase 1: Lustre
echo "🧪 PHASE 1: Remote Lustre" >> "$LOG"
singularity exec --nv --no-home --bind "/leonardo_scratch:/leonardo_scratch" "$SIF" \
    python3 -u "$L_TMP/reader_probe.py" "$L_TMP/wav_files" "$L_TMP/dataset.h5" "REMOTE" "$N" >> "$LOG" 2>&1

# Fase 2: Staging
echo "🧪 PHASE 2: Staging-In" >> "$LOG"
LOCAL_SSD="$LOCAL_SCRATCH/bench"
mkdir -p "$LOCAL_SSD/wavs"
t1=$(date +%s.%N)
cp "$L_TMP/wav_files/"*.wav "$LOCAL_SSD/wavs/" >> "$LOG" 2>&1
echo "  - CP WAVs: $(python3 -c "print(f'{($(date +%s.%N) - $t1):.4f}')")s" >> "$LOG"
cp "$L_TMP/dataset.h5" "$LOCAL_SSD/" >> "$LOG" 2>&1

# Fase 3: SSD
echo "🧪 PHASE 3: Local SSD" >> "$LOG"
singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$LOCAL_SCRATCH:$LOCAL_SCRATCH" \
    "$SIF" \
    python3 -u "$L_TMP/reader_probe.py" "$LOCAL_SSD/wavs" "$LOCAL_SSD/dataset.h5" "LOCAL" "$N" >> "$LOG" 2>&1

echo "🏁 NODE FINISH: $(date)" >> "$LOG"
EOF

# --- 4. SUBMISSION ---
echo "📤 Submitting..."
JOB_ID=$(sbatch --parsable "${LUSTRE_TMP}/io_test_slurm.sh")
tail -f "$STREAM_LOG" &
TAIL_PID=$!

# Monitoraggio più lento per dare tempo al filesystem di scrivere
while true; do
    STATE=$(sacct -j "$JOB_ID" --format=State --noheader | head -n 1 | xargs)
    if [[ "$STATE" == "COMPLETED" || "$STATE" == "FAILED" || "$STATE" == "TIMEOUT" ]]; then
        sleep 5 # Aspettiamo 5 secondi extra per il flush dei log
        break
    fi
    sleep 5
done

kill $TAIL_PID 2>/dev/null
echo -e "\n✅ Fine."

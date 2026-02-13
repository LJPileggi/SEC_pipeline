#!/bin/bash

# --- 1. CONFIGURATION (STRICTLY ABSOLUTE) ---
USER_NAME=$(whoami)
PROJECT_DIR="/leonardo_scratch/large/userexternal/${USER_NAME}/SEC_pipeline"
LUSTRE_TMP="${PROJECT_DIR}/.tmp_io_bench"
STREAM_LOG="${LUSTRE_TMP}/io_results.log"
SIF_FILE="${PROJECT_DIR}/.containers/clap_pipeline.sif"

# Cleanup and setup
rm -rf "$LUSTRE_TMP"
mkdir -p "${LUSTRE_TMP}/wav_files"
touch "$STREAM_LOG"

# --- 2. GENERATION PHASE ---
echo "🔨 Phase 0: Generating data..."
# Passiamo i parametri come costanti per non sbagliare
singularity exec --no-home --bind "${LUSTRE_TMP}:/mnt_lustre" "$SIF_FILE" \
    python3 -u - <<'PY'
import numpy as np
import soundfile as sf
import h5py
import os
wav_dir, h5_p = "/mnt_lustre/wav_files", "/mnt_lustre/dataset.h5"
n, sr, d = 500, 44100, 1
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
# IMPORTANTE: Usiamo 'EOF' tra apici singoli per disabilitare QUALSIASI espansione lato login
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

# Recupero dinamico dei percorsi sul nodo
NODE_LUSTRE_TMP=$(dirname $(realpath $0))
NODE_LOG="${NODE_LUSTRE_TMP}/io_results.log"
NODE_SIF="/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.containers/clap_pipeline.sif"

# Fallback robusto per SSD locale
NODE_SSD="${LOCAL_SCRATCH:-/scratch_local}/bench_${SLURM_JOB_ID}"
mkdir -p "${NODE_SSD}/wavs"

echo "🚀 NODE START: $(date)" >> "$NODE_LOG"

# --- Creazione Probe ---
cat << 'PY_INNER' > "${NODE_LUSTRE_TMP}/reader_probe.py"
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

# PHASE 1: Lustre
echo "🧪 PHASE 1: Remote Lustre" >> "$NODE_LOG"
singularity exec --nv --no-home --bind "/leonardo_scratch:/leonardo_scratch" "$NODE_SIF" \
    python3 -u "${NODE_LUSTRE_TMP}/reader_probe.py" "${NODE_LUSTRE_TMP}/wav_files" "${NODE_LUSTRE_TMP}/dataset.h5" "REMOTE" 500 >> "$NODE_LOG" 2>&1

# PHASE 2: Staging
echo "🧪 PHASE 2: Staging-In" >> "$NODE_LOG"
t1=$(date +%s.%N)
cp "${NODE_LUSTRE_TMP}/wav_files/"*.wav "${NODE_SSD}/wavs/" >> "$NODE_LOG" 2>&1
echo "  - CP WAVs: $(python3 -c "print(f'{($(date +%s.%N) - $t1):.4f}')")s" >> "$NODE_LOG"
cp "${NODE_LUSTRE_TMP}/dataset.h5" "${NODE_SSD}/dataset.h5" >> "$NODE_LOG" 2>&1

# PHASE 3: SSD
echo "🧪 PHASE 3: Local SSD" >> "$NODE_LOG"
singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "${NODE_SSD}:${NODE_SSD}" \
    "$NODE_SIF" \
    python3 -u "${NODE_LUSTRE_TMP}/reader_probe.py" "${NODE_SSD}/wavs" "${NODE_SSD}/dataset.h5" "LOCAL" 500 >> "$NODE_LOG" 2>&1

echo "🏁 NODE FINISH: $(date)" >> "$NODE_LOG"
EOF

# --- 4. EXECUTION ---
echo "📤 Submitting I/O job..."
JOB_ID=$(sbatch --parsable "${LUSTRE_TMP}/io_test_slurm.sh")

# Tail con controllo manuale
tail -f "$STREAM_LOG" &
TAIL_PID=$!

while true; do
    STATE=$(sacct -j "$JOB_ID" --format=State --noheader | head -n 1 | xargs)
    if [[ "$STATE" == "COMPLETED" || "$STATE" == "FAILED" || "$STATE" == "TIMEOUT" || "$STATE" == "CANCELLED" ]]; then
        sleep 5
        break
    fi
    sleep 2
done

kill $TAIL_PID 2>/dev/null
echo -e "\n✅ Script concluso."

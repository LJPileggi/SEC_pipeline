#!/bin/bash

# --- 1. CONFIGURATION (PERCORSI FISSI) ---
# Sostituisci lpilegg1 con il tuo utente se diverso, ma dai log precedenti sembra corretto
MY_USER="lpilegg1"
PROJECT_DIR="/leonardo_scratch/large/userexternal/${MY_USER}/SEC_pipeline"
LUSTRE_TMP="${PROJECT_DIR}/.tmp_io_bench"
STREAM_LOG="${LUSTRE_TMP}/io_results.log"
SIF_FILE="${PROJECT_DIR}/.containers/clap_pipeline.sif"

# Preparazione cartelle
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

# --- 3. GENERATE SLURM BATCH SCRIPT (HARDCODED) ---
# Usiamo 'EOF' tra singoli apici: lo script che arriva al nodo sarà ESATTAMENTE questo testo.
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

# PERCORSI ASSOLUTI (No variabili, no errori)
L_TMP="/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.tmp_io_bench"
LOG="/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.tmp_io_bench/io_results.log"
SIF="/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.containers/clap_pipeline.sif"
# Slurm fornisce LOCAL_SCRATCH, lo usiamo direttamente
LOCAL_DATA="${LOCAL_SCRATCH}/io_bench"
mkdir -p "${LOCAL_DATA}/wavs"

echo "🚀 NODE START: $(date)" >> "$LOG"

# --- Creazione Probe Python ---
cat << 'PY_PROBE' > "/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.tmp_io_bench/reader_probe.py"
import time, h5py, soundfile as sf, sys, os
def run(wav_p, h5_p, label):
    n = 500
    print(f"\n--- I/O Test: {label} ---", flush=True)
    try:
        # POSIX
        s = time.perf_counter()
        for i in range(n):
            with sf.SoundFile(os.path.join(wav_p, f"track_{i}.wav")) as f: _ = f.read()
        print(f"🔹 POSIX: {time.perf_counter() - s:.4f}s", flush=True)
        # HDF5
        s = time.perf_counter()
        with h5py.File(h5_p, 'r') as h5:
            ds = h5['audio']
            for i in range(n): _ = ds[i]
        print(f"🔹 HDF5:  {time.perf_counter() - s:.4f}s", flush=True)
    except Exception as e:
        print(f"❌ ERROR: {e}", flush=True)

if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], sys.argv[3])
PY_PROBE

# PHASE 1: Lustre
echo "🧪 PHASE 1: Remote Lustre" >> "$LOG"
singularity exec --nv --no-home --bind "/leonardo_scratch:/leonardo_scratch" "$SIF" \
    python3 -u "$L_TMP/reader_probe.py" "$L_TMP/wav_files" "$L_TMP/dataset.h5" "REMOTE" >> "$LOG" 2>&1

# PHASE 2: Staging
echo "🧪 PHASE 2: Staging-In" >> "$LOG"
t1=$(date +%s.%N)
cp "$L_TMP/wav_files/"*.wav "${LOCAL_DATA}/wavs/" >> "$LOG" 2>&1
echo "  - CP WAVs: $(python3 -c "print(f'{($(date +%s.%N) - $t1):.4f}')")s" >> "$LOG"
cp "$L_TMP/dataset.h5" "${LOCAL_DATA}/dataset.h5" >> "$LOG" 2>&1

# PHASE 3: Local SSD
echo "🧪 PHASE 3: Local SSD" >> "$LOG"
# Montiamo LOCAL_SCRATCH sulla stessa posizione interna
singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "${LOCAL_SCRATCH}:${LOCAL_SCRATCH}" \
    "$SIF" \
    python3 -u "$L_TMP/reader_probe.py" "${LOCAL_DATA}/wavs" "${LOCAL_DATA}/dataset.h5" "LOCAL" >> "$LOG" 2>&1

echo "🏁 NODE FINISH: $(date)" >> "$LOG"
EOF

# --- 4. SUBMISSION ---
echo "📤 Submitting..."
JOB_ID=$(sbatch --parsable "${LUSTRE_TMP}/io_test_slurm.sh")
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
echo -e "\n✅ Procedura completata."

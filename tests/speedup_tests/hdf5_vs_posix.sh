#!/bin/bash

# --- 1. CONFIGURATION ---
PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
LUSTRE_TMP="${PROJECT_DIR}/.tmp_io_bench"
SSD_HOST_DIR="/tmp/io_bench_$SLURM_JOB_ID"
STREAM_LOG="${LUSTRE_TMP}/io_results.log"
SIF_FILE="${PROJECT_DIR}/.containers/clap_pipeline.sif"

# Settings
N_FILES=500  # Ridotto per il test iniziale
SR=44100
DUR=1

cleanup() {
    echo -e "\n🧹 Cleanup phase..."
    if [ ! -z "$TAIL_PID" ]; then kill "$TAIL_PID" 2>/dev/null; fi
    # rm -rf "$LUSTRE_TMP" # Commentato temporaneamente per debuggare i file rimasti
}
trap cleanup EXIT SIGTERM SIGINT

mkdir -p "${LUSTRE_TMP}/wav_files"
touch "$STREAM_LOG"

# --- 2. GENERATION PHASE ---
echo "🔨 Phase 0: Generating data (Singularity)..."
# Usiamo variabili d'ambiente per passare i dati al python interno senza conflitti di escape
export N_FILES SR DUR
singularity exec --no-home \
    --bind "${LUSTRE_TMP}:/mnt_lustre" \
    "$SIF_FILE" \
    python3 -u - <<PY
import numpy as np
import soundfile as sf
import h5py
import os

n = int(os.environ['N_FILES'])
sr = int(os.environ['SR'])
d = int(os.environ['DUR'])
wav_dir = "/mnt_lustre/wav_files"
h5_p = "/mnt_lustre/dataset.h5"

print(f"Creating {n} files in {wav_dir}...")
data = np.random.uniform(-1, 1, sr * d).astype(np.float32)

with h5py.File(h5_p, 'w') as h5:
    ds = h5.create_dataset('audio', (n, sr*d), dtype='f4')
    for i in range(n):
        sf.write(f"{wav_dir}/track_{i}.wav", data, sr)
        ds[i] = data
print("✅ Python Generation Done.")
PY

# --- 3. GENERATE SLURM SCRIPT ---
# IMPORTANTE: Usiamo 'EOF' tra virgolette per evitare che il bash locale espanda le variabili del nodo
cat << 'EOF' > "${LUSTRE_TMP}/io_test_slurm.sh"
#!/bin/bash
#SBATCH --job-name=io_bench
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH -A IscrC_Pb-skite
#SBATCH --output=/dev/null
#SBATCH --error=/mnt_lustre/slurm_error.log # Catturiamo gli errori del nodo qui!

# Carichiamo i parametri passati dall'ambiente o hardcoded
N_FILES=500
STREAM_LOG="/mnt_lustre/io_results.log"

log_bench() { echo -e "$1" | tee -a "$STREAM_LOG"; }

echo "🚀 NODE: Starting Execution..." >> "$STREAM_LOG"

# --- Python Probe ---
cat << 'PY' > "/mnt_lustre/reader_probe.py"
import time
import h5py
import soundfile as sf
import sys
import os

wav_p, h5_p, label, n_files = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

print(f"\n--- Testing I/O: {label} (Samples: {n_files}) ---")
# POSIX
s = time.perf_counter()
for i in range(n_files):
    with sf.SoundFile(f"{wav_p}/track_{i}.wav") as f:
        _ = f.read()
print(f"🔹 POSIX: {time.perf_counter() - s:.4f}s")

# HDF5
s = time.perf_counter()
with h5py.File(h5_p, 'r') as h5:
    ds = h5['audio']
    for i in range(n_files):
        _ = ds[i]
print(f"🔹 HDF5:  {time.perf_counter() - s:.4f}s")
PY

# --- PHASE 1 ---
log_bench "🧪 PHASE 1: Remote Lustre"
singularity exec --nv --no-home --bind "/mnt_lustre:/mnt_lustre" "$SIF_FILE" \
    python3 -u /mnt_lustre/reader_probe.py "/mnt_lustre/wav_files" "/mnt_lustre/dataset.h5" "REMOTE" "$N_FILES" >> "$STREAM_LOG" 2>&1

# --- PHASE 2 ---
log_bench "🧪 PHASE 2: Staging-In"
mkdir -p "/mnt_ssd"
t1=$(date +%s.%N)
cp /mnt_lustre/wav_files/*.wav /mnt_ssd/
log_bench "  - CP WAVs: $(python3 -c "print(f'{($(date +%s.%N) - $t1):.4f}')")s"

t2=$(date +%s.%N)
cp /mnt_lustre/dataset.h5 /mnt_ssd/
log_bench "  - CP HDF5: $(python3 -c "print(f'{($(date +%s.%N) - $t2):.4f}')")s"

# --- PHASE 3 ---
log_bench "🧪 PHASE 3: Local SSD"
singularity exec --nv --no-home --bind "/mnt_lustre:/mnt_lustre" --bind "/mnt_ssd:/mnt_ssd" "$SIF_FILE" \
    python3 -u /mnt_lustre/reader_probe.py "/mnt_ssd" "/mnt_ssd/dataset.h5" "LOCAL" "$N_FILES" >> "$STREAM_LOG" 2>&1
EOF

# --- 4. EXECUTION ---
echo "📤 Submitting..."
# Inseriamo il mount nel comando sbatch tramite --bind globale se possibile, 
# ma qui lo gestiamo meglio dentro lo script
JOB_ID=$(sbatch --parsable "${LUSTRE_TMP}/io_test_slurm.sh")

echo "📊 Monitoring $JOB_ID..."
tail -f "$STREAM_LOG" &
TAIL_PID=$!

while sacct -j "$JOB_ID" --format=State --noheader | grep -qE "RUNNING|PENDING"; do sleep 1; done
sleep 2
kill $TAIL_PID 2>/dev/null

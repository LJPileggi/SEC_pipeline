#!/bin/bash

# ==============================================================================
# HPC I/O BENCHMARK: POSIX (MANY SMALL FILES) VS. HDF5 (SINGLE CONTAINER FILE)
#
# This script quantifies:
# 1. Staging latency (copying 1000 files vs 1 large file)
# 2. Access latency (opening/closing many files vs indexed HDF5 access)
# 3. Network vs Local performance (Lustre vs SSD)
# ==============================================================================

# --- 1. CONFIGURATION ---
PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
TMP_DIR="${PROJECT_DIR}/.tmp_io_bench"
LUSTRE_PATH="${TMP_DIR}/lustre"
SSD_PATH="/tmp/io_bench_$SLURM_JOB_ID" # Will be mapped to LOCAL_SCRATCH
STREAM_LOG="${TMP_DIR}/io_benchmark_results.log"

N_FILES=1000
SAMPLE_RATE=44100
DURATION=1 # seconds

cleanup() {
    echo -e "\n🧹 Cleaning up: $TMP_DIR and $SSD_PATH"
    rm -rf "$TMP_DIR"
    rm -rf "$SSD_PATH"
    if [ ! -z "$TAIL_PID" ]; then kill "$TAIL_PID" 2>/dev/null; fi
}
trap cleanup EXIT SIGTERM SIGINT

mkdir -p "$LUSTRE_PATH/wav_files"
touch "$STREAM_LOG"

# --- 2. GENERATION PHASE (Python Dummy Creator) ---
echo "🔨 Generating $N_FILES dummy tracks and HDF5 container..."
python3 -u <<PY
import numpy as np
import soundfile as sf
import h5py
import os
import time

n_files = $N_FILES
sr = $SAMPLE_RATE
dur = $DURATION
lustre_wav = "$LUSTRE_PATH/wav_files"
lustre_h5 = "$LUSTRE_PATH/dataset.h5"

# Generate dummy audio
data = np.random.uniform(-1, 1, sr * dur).astype(np.float32)

# Create HDF5
with h5py.File(lustre_h5, 'w') as h5:
    ds = h5.create_dataset('audio', (n_files, sr*dur), dtype='f4')
    for i in range(n_files):
        # Save individual WAV
        sf.write(f"{lustre_wav}/track_{i}.wav", data, sr)
        # Save into HDF5
        ds[i] = data
print("✅ Generation complete.")
PY

# --- 3. SLURM BATCH SCRIPT GENERATION ---
cat <<EOF > "${TMP_DIR}/io_test_slurm.sh"
#!/bin/bash
#SBATCH --job-name=io_bench
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null

log_bench() { echo -e "\$1" | tee -a "$STREAM_LOG"; }

# --- Python Reader Probe ---
cat <<PY > "${TMP_DIR}/reader_probe.py"
import time
import os
import h5py
import soundfile as sf
import sys

path_wav = sys.argv[1]
path_h5 = sys.argv[2]
label = sys.argv[3]

def run_bench():
    print(f"\n--- Testing I/O: {label} ---")
    
    # POSIX Test
    start = time.perf_counter()
    for i in range($N_FILES):
        with sf.SoundFile(f"{path_wav}/track_{i}.wav") as f:
            _ = f.read()
    t_posix = time.perf_counter() - start
    print(f"🔹 POSIX (Individual WAVs): {t_posix:.4f}s")

    # HDF5 Test
    start = time.perf_counter()
    with h5py.File(path_h5, 'r') as h5:
        ds = h5['audio']
        for i in range($N_FILES):
            _ = ds[i]
    t_h5 = time.perf_counter() - start
    print(f"🔹 HDF5 (Single File):      {t_h5:.4f}s")

if __name__ == '__main__':
    run_bench()
PY

# --- BENCHMARK 1: REMOTE I/O (LUSTRE) ---
log_bench "\n📡 PHASE 1: Remote I/O Performance (Directly on Lustre)"
python3 -u "${TMP_DIR}/reader_probe.py" "$LUSTRE_PATH/wav_files" "$LUSTRE_PATH/dataset.h5" "REMOTE_LUSTRE" >> "$STREAM_LOG" 2>&1

# --- BENCHMARK 2: STAGING SPEED (THE CORE OF YOUR THESIS) ---
log_bench "\n🚚 PHASE 2: Staging-In Throughput (Lustre -> Local SSD)"
mkdir -p "$SSD_PATH/wav_files"

start_wav=\$(date +%s.%N)
cp "$LUSTRE_PATH/wav_files/"*.wav "$SSD_PATH/wav_files/"
end_wav=\$(date +%s.%N)
log_bench "  - Staging WAV folder: \$(echo "\$end_wav - \$start_wav" | bc)s"

start_h5=\$(date +%s.%N)
cp "$LUSTRE_PATH/dataset.h5" "$SSD_PATH/"
end_h5=\$(date +%s.%N)
log_bench "  - Staging HDF5 file:   \$(echo "\$end_h5 - \$start_h5" | bc)s"

# --- BENCHMARK 3: LOCAL I/O (SSD) ---
log_bench "\n⚡ PHASE 3: Local I/O Performance (After Staging on SSD)"
python3 -u "${TMP_DIR}/reader_probe.py" "$SSD_PATH/wav_files" "$SSD_PATH/dataset.h5" "LOCAL_SSD" >> "$STREAM_LOG" 2>&1
EOF

# --- 4. EXECUTION ---
echo "📤 Submitting I/O Benchmark Job..."
JOB_ID=$(sbatch --parsable "${TMP_DIR}/io_test_slurm.sh")

echo "📊 STREAMING RESULTS (Job ID: $JOB_ID):"
tail -f "$STREAM_LOG" &
TAIL_PID=$!

while sacct -j "$JOB_ID" --format=State --noheader | grep -qE "RUNNING|PENDING"; do sleep 2; done
sleep 2
kill $TAIL_PID 2>/dev/null
echo -e "\n✅ Benchmark Complete."

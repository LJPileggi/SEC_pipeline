#!/bin/bash

# ==============================================================================
# HPC I/O BENCHMARK: POSIX VS. HDF5 (CONTAINERIZED VERSION)
#
# This script runs entirely within the Singularity container to ensure 
# library availability (h5py, soundfile, numpy). 
# It tests I/O across Lustre and Local SSD (/tmp).
# ==============================================================================

# --- 1. CONFIGURATION ---
PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
# Absolute path on Lustre (Remote)
LUSTRE_TMP="${PROJECT_DIR}/.tmp_io_bench"
# Absolute path on Local SSD (mapped to /tmp_ssd inside container)
SSD_HOST_DIR="/tmp/io_bench_$SLURM_JOB_ID"
STREAM_LOG="${LUSTRE_TMP}/io_results.log"

SIF_FILE="${PROJECT_DIR}/.containers/clap_pipeline.sif"

N_FILES=1000
SAMPLE_RATE=44100
DURATION=1 

# Cleanup logic
cleanup() {
    echo -e "\n🧹 Cleaning up benchmark directories..."
    rm -rf "$LUSTRE_TMP"
    rm -rf "$SSD_HOST_DIR"
    if [ ! -z "$TAIL_PID" ]; then kill "$TAIL_PID" 2>/dev/null; fi
}
trap cleanup EXIT SIGTERM SIGINT

mkdir -p "${LUSTRE_TMP}/wav_files"
mkdir -p "$SSD_HOST_DIR"
touch "$STREAM_LOG"

# --- 2. GENERATION PHASE (Inside Container) ---
echo "🔨 Generating dummy data inside Container..."
# We bind the Lustre tmp dir to /mnt_lustre inside the container
singularity exec --no-home \
    --bind "${LUSTRE_TMP}:/mnt_lustre" \
    "$SIF_FILE" \
    python3 -u <<PY
import numpy as np
import soundfile as sf
import h5py
import os

n_files = $N_FILES
sr = $SAMPLE_RATE
dur = $DURATION
wav_dir = "/mnt_lustre/wav_files"
h5_path = "/mnt_lustre/dataset.h5"

data = np.random.uniform(-1, 1, sr * dur).astype(np.float32)

with h5py.File(h5_path, 'w') as h5:
    ds = h5.create_dataset('audio', (n_files, sr*dur), dtype='f4')
    for i in range(n_files):
        sf.write(f"{wav_dir}/track_{i}.wav", data, sr)
        ds[i] = data
print("✅ Generation complete.")
PY

# --- 3. SLURM BATCH SCRIPT GENERATION ---
cat <<EOF > "${LUSTRE_TMP}/io_test_slurm.sh"
#!/bin/bash
#SBATCH --job-name=io_bench
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH -A IscrC_Pb-skite
#SBATCH --output=/dev/null

log_bench() { echo -e "\$1" | tee -a "$STREAM_LOG"; }

# --- Python Reader Probe ---
cat <<PY > "${LUSTRE_TMP}/reader_probe.py"
import time
import h5py
import soundfile as sf
import sys

# Paths inside the container
wav_path = sys.argv[1]
h5_path = sys.argv[2]
label = sys.argv[3]

print(f"\n--- Testing I/O: {label} ---")
# Test POSIX
start = time.perf_counter()
for i in range($N_FILES):
    with sf.SoundFile(f"{wav_path}/track_{i}.wav") as f:
        _ = f.read()
print(f"🔹 POSIX (Individual WAVs): {time.perf_counter() - start:.4f}s")

# Test HDF5
start = time.perf_counter()
with h5py.File(h5_path, 'r') as h5:
    ds = h5['audio']
    for i in range($N_FILES):
        _ = ds[i]
print(f"🔹 HDF5 (Single File):      {time.perf_counter() - start:.4f}s")
PY

# Helper for container execution
run_in_container() {
    local label=\$1
    local wav_in_cont=\$2
    local h5_in_cont=\$3
    
    singularity exec --nv --no-home \\
        --bind "${LUSTRE_TMP}:/mnt_lustre" \\
        --bind "${SSD_HOST_DIR}:/mnt_ssd" \\
        "$SIF_FILE" \\
        python3 -u /mnt_lustre/reader_probe.py "\$wav_in_cont" "\$h5_in_cont" "\$label" >> "$STREAM_LOG" 2>&1
}

log_bench "\n📡 PHASE 1: Remote I/O Performance (Directly on Lustre)"
run_in_container "REMOTE_LUSTRE" "/mnt_lustre/wav_files" "/mnt_lustre/dataset.h5"

log_bench "\n🚚 PHASE 2: Staging-In Throughput (Lustre -> Local SSD)"
# Measured outside container (standard cp)
start_wav=\$(date +%s.%N)
cp "${LUSTRE_TMP}/wav_files/"*.wav "${SSD_HOST_DIR}/"
log_bench "  - Staging WAVs: \$(python3 -c "print(f'{($(date +%s.%N) - \$start_wav):.4f}')")s"

start_h5=\$(date +%s.%N)
cp "${LUSTRE_TMP}/dataset.h5" "${SSD_HOST_DIR}/"
log_bench "  - Staging HDF5: \$(python3 -c "print(f'{($(date +%s.%N) - \$start_h5):.4f}')")s"

log_bench "\n⚡ PHASE 3: Local I/O Performance (On SSD)"
run_in_container "LOCAL_SSD" "/mnt_ssd" "/mnt_ssd/dataset.h5"
EOF

# --- 4. EXECUTION ---
echo "📤 Submitting I/O Benchmark Job..."
JOB_ID=$(sbatch --parsable "${LUSTRE_TMP}/io_test_slurm.sh")

echo "📊 MONITORING I/O RESULTS (Job ID: $JOB_ID):"
tail -f "$STREAM_LOG" &
TAIL_PID=$!

while sacct -j "$JOB_ID" --format=State --noheader | grep -qE "RUNNING|PENDING"; do sleep 2; done
sleep 2
kill $TAIL_PID 2>/dev/null
echo -e "\n✅ Benchmark Complete."

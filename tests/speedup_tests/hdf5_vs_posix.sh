#!/bin/bash

# --- 1. CONFIGURATION ---
MY_USER="lpilegg1"
PROJECT_DIR="/leonardo_scratch/large/userexternal/${MY_USER}/SEC_pipeline"
L_TMP="${PROJECT_DIR}/.tmp_io_final"
STREAM_LOG="${L_TMP}/benchmark_live.log"
SIF="/leonardo_scratch/large/userexternal/${MY_USER}/SEC_pipeline/.containers/clap_pipeline.sif"

# Parametri statistici
ITERATIONS=10
BATCH_SIZE=500
TOTAL_FILES=$((ITERATIONS * BATCH_SIZE))

rm -rf "$L_TMP"
mkdir -p "${L_TMP}/wav_files"
touch "$STREAM_LOG"

# --- 2. GENERATION (5000 file) ---
echo "🔨 Generazione dataset di test ($TOTAL_FILES tracce)..."
singularity exec --no-home --bind "${L_TMP}:/mnt_lustre" "$SIF" \
    python3 -u - <<'PY'
import numpy as np
import soundfile as sf
import h5py
import os
n, sr, d = 5000, 44100, 1
wav_dir, h5_p = "/mnt_lustre/wav_files", "/mnt_lustre/dataset.h5"
data = np.random.uniform(-1, 1, sr * d).astype(np.float32)
with h5py.File(h5_p, 'w') as h5:
    ds = h5.create_dataset('audio', (n, sr*d), dtype='f4')
    for i in range(n):
        sf.write(f"{wav_dir}/track_{i}.wav", data, sr)
        ds[i] = data
PY

# --- 3. SLURM BATCH SCRIPT ---
cat << 'EOF' > "${L_TMP}/run_ultimate.sh"
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH -A IscrC_Pb-skite
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

L_TMP="/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.tmp_io_final"
LOG="${L_TMP}/benchmark_live.log"
SIF="/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.containers/clap_pipeline.sif"
SSD_PATH="/tmp/bench_${SLURM_JOB_ID}"

echo "🚀 NODE START: $(date)" >> "$LOG"

# --- Python Probe ---
cat << 'PY_PROBE' > "/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.tmp_io_final/probe.py"
import time, h5py, soundfile as sf, sys, os
def run_test(wav_p, h5_p, start, count, label):
    # POSIX
    ts = os.times(); ws = time.perf_counter()
    for i in range(start, start + count):
        with sf.SoundFile(os.path.join(wav_p, f"track_{i}.wav")) as f: _ = f.read()
    te = os.times(); we = time.perf_counter()
    p_w, p_s = we - ws, te[1] - ts[1]
    
    # HDF5
    ts = os.times(); ws = time.perf_counter()
    with h5py.File(h5_p, 'r') as h5:
        ds = h5['audio']
        for i in range(start, start + count): _ = ds[i]
    te = os.times(); we = time.perf_counter()
    h_w, h_s = we - ws, te[1] - ts[1]
    
    print(f"{label}|{p_w:.6f}|{p_s:.6f}|{h_w:.6f}|{h_s:.6f}", flush=True)

run_test(sys.argv[1], sys.argv[2], int(sys.argv[3]), 500, sys.argv[4])
PY_PROBE

# HEADER
echo "Phase|POSIX_Wall|POSIX_Sys|HDF5_Wall|HDF5_Sys" >> "$LOG"

for i in {0..9}; do
    OFFSET=$((i * 500))
    echo "🧪 ITERATION $i (LUSTRE)..." >> "$LOG"
    singularity exec --no-home --bind "/leonardo_scratch:/leonardo_scratch" "$SIF" \
        python3 -u "$L_TMP/probe.py" "$L_TMP/wav_files" "$L_TMP/dataset.h5" "$OFFSET" "LUSTRE_IT$i" >> "$LOG" 2>&1
done

echo "🧪 STAGING IN PROGRESS..." >> "$LOG"
# Misura Staging (una tantum per semplicità o nel loop)
mkdir -p "$SSD_PATH/wavs"
t_s=$(date +%s.%N)
cp "$L_TMP/wav_files/"*.wav "$SSD_PATH/wavs/"
cp "$L_TMP/dataset.h5" "$SSD_PATH/"
t_e=$(date +%s.%N)
echo "STAGING_TIME|$(python3 -c "print($t_e - $t_s)")" >> "$LOG"

for i in {0..9}; do
    OFFSET=$((i * 500))
    echo "🧪 ITERATION $i (SSD)..." >> "$LOG"
    singularity exec --no-home --bind "/tmp:/tmp" --bind "/leonardo_scratch:/leonardo_scratch" "$SIF" \
        python3 -u "$L_TMP/probe.py" "$SSD_PATH/wavs" "$SSD_PATH/dataset.h5" "$OFFSET" "SSD_IT$i" >> "$LOG" 2>&1
done

echo "🏁 NODE FINISH: $(date)" >> "$LOG"
EOF

# --- 4. EXECUTION & LIVE LOG ---
echo "📤 Submitting Job..."
JOB_ID=$(sbatch --parsable "${L_TMP}/run_ultimate.sh")

# RIPRISTINATO: Tail live per vedere i risultati mentre arrivano
tail -f "$STREAM_LOG" &
TAIL_PID=$!

while true; do
    STATE=$(sacct -j "$JOB_ID" --format=State --noheader | head -n 1 | xargs)
    if [[ "$STATE" == "COMPLETED" || "$STATE" == "FAILED" || "$STATE" == "TIMEOUT" ]]; then
        sleep 5
        break
    fi
    sleep 5
done

kill $TAIL_PID 2>/dev/null
echo -e "\n✅ Procedura conclusa. I dati sono in $STREAM_LOG"

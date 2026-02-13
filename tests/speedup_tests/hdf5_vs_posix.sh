#!/bin/bash

# --- 1. CONFIGURATION ---
PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
LUSTRE_TMP="${PROJECT_DIR}/.tmp_io_bench"
STREAM_LOG="${LUSTRE_TMP}/io_results.log"
SIF_FILE="${PROJECT_DIR}/.containers/clap_pipeline.sif"

export N_FILES=500
export SR=44100
export DUR=1

# Clean log and temp dir before starting
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
PY

# --- 3. GENERATE SLURM BATCH SCRIPT ---
cat << EOF > "${LUSTRE_TMP}/io_test_slurm.sh"
#!/bin/bash
#SBATCH --job-name=io_bench
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH -A IscrC_Pb-skite
#SBATCH --output=/dev/null
#SBATCH --error=${LUSTRE_TMP}/slurm_debug.err

# Identifichiamo la cartella SSD locale (usiamo una sottocartella di $LOCAL_SCRATCH)
LOCAL_STORAGE="\$LOCAL_SCRATCH/bench_data"
mkdir -p "\$LOCAL_STORAGE/wav_files"

echo "🚀 NODE START: \$(date)" >> "$STREAM_LOG"

# --- Python Probe (Updated to use /mnt_ssd fixed path) ---
cat << 'PY_INNER' > "${LUSTRE_TMP}/reader_probe.py"
import time, h5py, soundfile as sf, sys, os
wav_p, h5_p, label, n_files = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
print(f"\n--- I/O Test: {label} ---", flush=True)

# POSIX Test
s = time.perf_counter()
for i in range(n_files):
    f_path = os.path.join(wav_p, f"track_{i}.wav")
    with sf.SoundFile(f_path) as f: _ = f.read()
print(f"🔹 POSIX: {time.perf_counter() - s:.4f}s", flush=True)

# HDF5 Test
s = time.perf_counter()
with h5py.File(h5_p, 'r') as h5:
    ds = h5['audio']
    for i in range(n_files): _ = ds[i]
print(f"🔹 HDF5:  {time.perf_counter() - s:.4f}s", flush=True)
PY_INNER

# PHASE 1: Remote Lustre
echo "🧪 PHASE 1: Remote Lustre" >> "$STREAM_LOG"
singularity exec --nv --no-home --bind "${LUSTRE_TMP}:/mnt_lustre" "$SIF_FILE" \\
    python3 -u /mnt_lustre/reader_probe.py "/mnt_lustre/wav_files" "/mnt_lustre/dataset.h5" "REMOTE_LUSTRE" "$N_FILES" >> "$STREAM_LOG" 2>&1

# PHASE 2: Staging
echo "🧪 PHASE 2: Staging-In" >> "$STREAM_LOG"
t1=\$(date +%s.%N)
cp ${LUSTRE_TMP}/wav_files/*.wav "\$LOCAL_STORAGE/wav_files/"
echo "  - CP WAVs: \$(python3 -c "print(f'{(\$(date +%s.%N) - \$t1):.4f}')")s" >> "$STREAM_LOG"

t2=\$(date +%s.%N)
cp ${LUSTRE_TMP}/dataset.h5 "\$LOCAL_STORAGE/"
echo "  - CP HDF5: \$(python3 -c "print(f'{(\$(date +%s.%N) - \$t2):.4f}')")s" >> "$STREAM_LOG"

sync && sleep 1

# PHASE 3: Local SSD (Force mount on /mnt_ssd)
echo "🧪 PHASE 3: Local SSD" >> "$STREAM_LOG"
singularity exec --nv --no-home \\
    --bind "${LUSTRE_TMP}:/mnt_lustre" \\
    --bind "\$LOCAL_STORAGE:/mnt_ssd" \\
    "$SIF_FILE" \\
    python3 -u /mnt_lustre/reader_probe.py "/mnt_ssd/wav_files" "/mnt_ssd/dataset.h5" "LOCAL_SSD" "$N_FILES" >> "$STREAM_LOG" 2>&1

echo "🏁 NODE FINISH: \$(date)" >> "$STREAM_LOG"
EOF

# --- 4. SUBMISSION ---
echo "📤 Submitting I/O job..."
JOB_ID=$(sbatch --parsable "${LUSTRE_TMP}/io_test_slurm.sh")

echo "📊 Monitoring Job $JOB_ID..."
tail -f "$STREAM_LOG" &
TAIL_PID=$!

while sacct -j "$JOB_ID" --format=State --noheader | grep -qE "RUNNING|PENDING|COMPLETING"; do sleep 2; done
sleep 2
kill $TAIL_PID 2>/dev/null
echo -e "\n✅ Done."

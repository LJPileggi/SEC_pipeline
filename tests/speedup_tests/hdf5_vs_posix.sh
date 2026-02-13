#!/bin/bash

# --- 1. CONFIGURATION ---
MY_USER=$(whoami)
PROJECT_DIR="/leonardo_scratch/large/userexternal/${MY_USER}/SEC_pipeline"
L_TMP="${PROJECT_DIR}/.tmp_io_final"
RAW_DATA="${L_TMP}/raw_results.csv"
SIF="${PROJECT_DIR}/.containers/clap_pipeline.sif"

ITERATIONS=10
BATCH_SIZE=500

rm -rf "$L_TMP"
mkdir -p "${L_TMP}/wav_files"

# --- 2. GENERATION (5000 file) ---
echo "🔨 Phase 0: Generazione Dataset (5000 file)..."
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

# --- 3. SLURM SCRIPT (CORRETTO) ---
cat << 'EOF' > "${L_TMP}/run_final.sh"
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH -A IscrC_Pb-skite
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

L_TMP="/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.tmp_io_final"
CSV="${L_TMP}/raw_results.csv"
SIF="/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.containers/clap_pipeline.sif"
SSD_BASE="/tmp/bench_${SLURM_JOB_ID}"

echo "Phase,Format,Iteration,Wall,Sys" > "$CSV"

# --- Probe (Output rigoroso) ---
cat << 'PY_PROBE' > "${L_TMP}/probe.py"
import time, h5py, soundfile as sf, sys, os
wav_p, h5_p, label, fmt, start = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
try:
    ts = os.times(); ws = time.perf_counter()
    if fmt == "POSIX":
        for i in range(start, start + 500):
            with sf.SoundFile(os.path.join(wav_p, f"track_{i}.wav")) as f: _ = f.read()
    else:
        with h5py.File(h5_p, 'r') as h5:
            ds = h5['audio']
            for i in range(start, start + 500): _ = ds[i]
    te = os.times(); we = time.perf_counter()
    print(f"{label},{fmt},{start//500},{we-ws:.6f},{te[1]-ts[1]:.6f}")
except Exception:
    pass # Evitiamo di sporcare il CSV se un file manca
PY_PROBE

# FASE 1: LUSTRE
for i in {0..9}; do
    OFFSET=$((i * 500))
    singularity exec --no-home --bind "/leonardo_scratch:/leonardo_scratch" "$SIF" \
        python3 -u "${L_TMP}/probe.py" "${L_TMP}/wav_files" "${L_TMP}/dataset.h5" "LUSTRE" "POSIX" "$OFFSET" >> "$CSV"
    singularity exec --no-home --bind "/leonardo_scratch:/leonardo_scratch" "$SIF" \
        python3 -u "${L_TMP}/probe.py" "${L_TMP}/wav_files" "${L_TMP}/dataset.h5" "LUSTRE" "HDF5" "$OFFSET" >> "$CSV"
done

# FASE 2 & 3: STAGING E SSD
for i in {0..9}; do
    OFFSET=$((i * 500))
    ITER_SSD="${SSD_BASE}_$i"
    mkdir -p "$ITER_SSD/wavs"
    
    # FIX: Staging POSIX (Uso di seq per risolvere il bug dell'espansione)
    t_s=$(date +%s.%N)
    for j in $(seq $OFFSET $((OFFSET + 499))); do
        cp "${L_TMP}/wav_files/track_${j}.wav" "$ITER_SSD/wavs/"
    done
    t_e=$(date +%s.%N)
    echo "STAGING,POSIX,$i,$(python3 -c "print($t_e - $t_s)"),0" >> "$CSV"
    
    # Staging HDF5
    t_s=$(date +%s.%N)
    cp "${L_TMP}/dataset.h5" "$ITER_SSD/data.h5"
    t_e=$(date +%s.%N)
    echo "STAGING,HDF5,$i,$(python3 -c "print($t_e - $t_s)"),0" >> "$CSV"

    # Test SSD
    singularity exec --no-home --bind "/tmp:/tmp" --bind "/leonardo_scratch:/leonardo_scratch" "$SIF" \
        python3 -u "${L_TMP}/probe.py" "$ITER_SSD/wavs" "$ITER_SSD/data.h5" "SSD" "POSIX" "$OFFSET" >> "$CSV"
    singularity exec --no-home --bind "/tmp:/tmp" --bind "/leonardo_scratch:/leonardo_scratch" "$SIF" \
        python3 -u "${L_TMP}/probe.py" "$ITER_SSD/wavs" "$ITER_SSD/data.h5" "SSD" "HDF5" "$OFFSET" >> "$CSV"
    
    rm -rf "$ITER_SSD"
done
EOF

# --- 4. ESECUZIONE E PARSING ---
echo "📤 Invio Job..."
JOB_ID=$(sbatch --parsable "${L_TMP}/run_final.sh")
while squeue -j "$JOB_ID" | grep -q "$JOB_ID"; do echo -n "."; sleep 5; done

echo -e "\n📊 Elaborazione Risultati..."
singularity exec --no-home --bind "${L_TMP}:${L_TMP}" "$SIF" python3 -u - <<PY_STATS
import pandas as pd
import numpy as np
df = pd.read_csv("${RAW_DATA}")
print("\n" + "="*95)
print(f"{'BENCHMARK FINALE: POSIX VS HDF5 (N=10)':^95}")
print("="*95)
print(f"{'FASE':<12} | {'FORMATO':<8} | {'WALL TIME (Media ± Dev.Std)':<30} | {'OVERHEAD SYS'}")
print("-" * 95)

for phase in ['LUSTRE', 'STAGING', 'SSD']:
    for fmt in ['POSIX', 'HDF5']:
        sub = df[(df['Phase'] == phase) & (df['Format'] == fmt)]
        if len(sub) > 0:
            m, s = sub['Wall'].mean(), sub['Wall'].std()
            sys_m = sub['Sys'].mean()
            ov = (sys_m / m * 100) if m > 0 else 0
            # Se la std è NaN (un solo campione), mettiamo 0.0
            s_val = s if not np.isnan(s) else 0.0
            print(f"{phase:<12} | {fmt:<8} | {m:>8.4f}s ± {s_val:.4f} | {ov:>6.1f}%")
        else:
            print(f"{phase:<12} | {fmt:<8} | {'MISSING DATA':<30} | {'---'}")
print("="*95)
PY_STATS

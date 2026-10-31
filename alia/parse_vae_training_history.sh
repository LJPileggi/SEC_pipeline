#!/bin/bash
# ==============================================================================
# inspect_vae_training_interactive.sh
# Esecuzione INTERATTIVA dentro Singularity (NO SLURM)
# Utilizzo:
#   chmod +x inspect_vae_training_interactive.sh
#   ./inspect_vae_training_interactive.sh [DIR_CHECKPOINT] [FILE_LOG_OUT]
# ==============================================================================

PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
SIF_FILE="$PROJECT_DIR/.containers/clap_pipeline.sif"
DEFAULT_MODELS_DIR="$PROJECT_DIR/.models/vae_model"

CKPT_DIR="${1:-$DEFAULT_MODELS_DIR}"
LOG_FILE="${2:-}"

if [ ! -f "$SIF_FILE" ]; then
    echo "❌ [ERRORE] File container non trovato: $SIF_FILE"
    exit 1
fi

if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ [ERRORE] Directory di progetto non trovata: $PROJECT_DIR"
    exit 1
fi

echo "========================================================================"
echo "🐳 Esecuzione Interattiva dentro Container Apptainer/Singularity"
echo "   • Container:   $SIF_FILE"
echo "   • Project Dir: $PROJECT_DIR"
echo "   • Checkpoints: $CKPT_DIR"
echo "   • Log File:    ${LOG_FILE:-'Auto-detect log più recente'}"
echo "========================================================================"

singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$PROJECT_DIR:/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 - "$CKPT_DIR" "$LOG_FILE" << 'EOF'
import os
import sys
import glob
import re
import torch

ckpt_dir = sys.argv[1]
log_file = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None

print("\n" + "=" * 80)
print(f"📦 1. ANALISI CHECKPOINT SU DISCO: {ckpt_dir}")
print("=" * 80)

ckpt_pattern = os.path.join(ckpt_dir, "dual_vae_epoch_*.pt")
ckpts = glob.glob(ckpt_pattern)

def extract_epoch(fname):
    m = re.search(r"dual_vae_epoch_(\d+)\.pt", fname)
    return int(m.group(1)) if m else -1

ckpts = sorted(ckpts, key=extract_epoch)

if not ckpts:
    print(f"⚠️ Nessun checkpoint trovato in: {ckpt_pattern}")
else:
    print(f"Trovati {len(ckpts)} checkpoint salvati:")
    print(f"{'Nome File':<28} | {'Epoca':<6} | {'Avg G-Loss':<11} | {'Commitment':<10} | {'CosAlign':<10}")
    print("-" * 80)
    for ckpt_path in ckpts:
        fname = os.path.basename(ckpt_path)
        try:
            data = torch.load(ckpt_path, map_location="cpu")
            ep = data.get("epoch", extract_epoch(fname))
            loss = data.get("loss", float("nan"))
            comm = data.get("commitment_cost", "N/A")
            cos_w = data.get("cos_align_weight", "N/A")
            print(f"{fname:<28} | {ep:<6} | {loss:<11.4f} | {str(comm):<10} | {str(cos_w):<10}")
        except Exception as e:
            print(f"{fname:<28} | ERRORE LETTURA: {e}")

print("\n" + "=" * 80)
print("📈 2. CRONOLOGIA METRICHE COMPLETA (RecM, RecOct, Cross, Cos, VQ, L1, Cls)")
print("=" * 80)

if not log_file or not os.path.exists(log_file):
    candidates = glob.glob("dual_vae_train_*.out") + glob.glob("*.out") + glob.glob("*.err")
    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        log_file = candidates[0]
        print(f"🔍 Rilevato log più recente: {log_file}")
    else:
        log_file = None

if not log_file or not os.path.exists(log_file):
    print("⚠️ Nessun file .out rilevato automaticamente.")
    print("   Specificare il file esplicito: ./inspect_vae_training_interactive.sh <DIR_CKPT> <FILE_LOG.out>")
    sys.exit(0)

line_pattern = re.compile(
    r"Ep\s+\[(\d+)\].*?\|\s+G:\s+([\d\.]+)\s+\|\s+\[RecM:\s+([\d\.]+)\s+\|\s+RecOct:\s+([\d\.]+)\s+\|\s+Cross:\s+([\d\.]+)\s+\|\s+Cos:\s+([\d\.]+)\s+\|\s+VQ:\s+([\d\.]+)\s+\|\s+Adv:\s+([\-\d\.]+)\]\s+\|\s+Cls:\s+([\d\.]+)\s+\|\s+L1:\s+([\d\.]+)%\s+\|\s+\|\|g_oct\|\|:\s+([\d\.]+)\s+\|\s+LR:\s+([\d\.e\-\+]+)"
)

epoch_data = {}

with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = line_pattern.search(line)
        if m:
            ep = int(m.group(1))
            g_loss = float(m.group(2))
            recm = float(m.group(3))
            recoct = float(m.group(4))
            cross = float(m.group(5))
            cos_val = float(m.group(6))
            vq_val = float(m.group(7))
            adv = float(m.group(8))
            cls_val = float(m.group(9))
            l1 = float(m.group(10))
            g_norm = float(m.group(11))
            lr = float(m.group(12))

            if ep not in epoch_data:
                epoch_data[ep] = {
                    "count": 0, "G": 0.0, "RecM": 0.0, "RecOct": 0.0, "Cross": 0.0,
                    "Cos": 0.0, "VQ": 0.0, "Adv": 0.0, "Cls": 0.0, "L1": 0.0, "grad": 0.0, "LR": lr
                }
            s = epoch_data[ep]
            s["count"] += 1
            s["G"] += g_loss
            s["RecM"] += recm
            s["RecOct"] += recoct
            s["Cross"] += cross
            s["Cos"] += cos_val
            s["VQ"] += vq_val
            s["Adv"] += adv
            s["Cls"] += cls_val
            s["L1"] += l1
            s["grad"] += g_norm
            s["LR"] = lr

if not epoch_data:
    print(f"⚠️ Nessuna riga di log parsabile trovata in {log_file}")
else:
    print(f"{'Ep':<4} | {'G-Loss':<7} | {'RecM':<6} | {'RecOct':<6} | {'Cross':<6} | {'Cos':<6} | {'VQ':<6} | {'L1-Match':<8} | {'Cls':<6} | {'||g_oct||':<8} | {'LR':<8}")
    print("-" * 92)
    for ep in sorted(epoch_data.keys()):
        s = epoch_data[ep]
        c = max(1, s["count"])
        print(
            f"{ep:<4d} | {s['G']/c:<7.3f} | {s['RecM']/c:<6.3f} | {s['RecOct']/c:<6.3f} | "
            f"{s['Cross']/c:<6.3f} | {s['Cos']/c:<6.3f} | {s['VQ']/c:<6.3f} | "
            f"{s['L1']/c:<7.1f}% | {s['Cls']/c:<6.3f} | {s['grad']/c:<8.2f} | {s['LR']:<8.1e}"
        )

print("=" * 80)
EOF

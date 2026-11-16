#!/bin/bash
# ==============================================================================
# inspect_vae_training_interactive.sh
# Esecuzione INTERATTIVA dentro Singularity (NO SLURM)
# Utilizzo:
#   chmod +x inspect_vae_training_interactive.sh
#   ./inspect_vae_training_interactive.sh [DIR_CHECKPOINT] [FILE_LOG_OUT] [MAX_EPOCH]
#
# Esempi:
#   ./inspect_vae_training_interactive.sh
#   ./inspect_vae_training_interactive.sh "" "" 44
#   ./inspect_vae_training_interactive.sh /tmp_data/models/vae_model slurm-12345.out 44
# ==============================================================================

PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
SIF_FILE="$PROJECT_DIR/.containers/clap_pipeline.sif"
DEFAULT_MODELS_DIR="$PROJECT_DIR/.models/vae_model"

MAX_EPOCH="${1:-}"
CKPT_DIR="${2:-$DEFAULT_MODELS_DIR}"
LOG_FILE="${3:-}"

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
echo "   • Max Epoch:   ${MAX_EPOCH:-'Tutte le epoche disponibili'}"
echo "========================================================================"

singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$PROJECT_DIR:/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 - "$CKPT_DIR" "$LOG_FILE" "$MAX_EPOCH" << 'EOF'
import os
import sys
import glob
import re
import torch

ckpt_dir = sys.argv[1]
log_file = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
max_epoch = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].strip() != "" else None

print("\n" + "=" * 96)
print(f"📦 1. ANALISI CHECKPOINT SU DISCO: {ckpt_dir}" + (f" (fino a Epoca {max_epoch})" if max_epoch is not None else ""))
print("=" * 96)

ckpt_pattern = os.path.join(ckpt_dir, "dual_vae_epoch_*.pt")
ckpts = glob.glob(ckpt_pattern)

def extract_epoch(fname):
    m = re.search(r"dual_vae_epoch_(\d+)\.pt", fname)
    return int(m.group(1)) if m else -1

ckpts = sorted(ckpts, key=extract_epoch)
if max_epoch is not None:
    ckpts = [c for c in ckpts if extract_epoch(c) <= max_epoch]

if not ckpts:
    print(f"⚠️ Nessun checkpoint trovato in: {ckpt_pattern}")
else:
    print(f"Trovati {len(ckpts)} checkpoint salvati:")
    print(f"{'Nome File':<26} | {'Ep':<4} | {'G-Loss':<8} | {'RecM':<7} | {'RecOct':<7} | {'Cross':<7} | {'Cos':<7} | {'Dist':<7} | {'VQ':<7} | {'Cls':<7} | {'FM':<7}")
    print("-" * 116)
    for ckpt_path in ckpts:
        fname = os.path.basename(ckpt_path)
        try:
            data = torch.load(ckpt_path, map_location="cpu")
            ep = data.get("epoch", extract_epoch(fname))
            loss = data.get("loss", float("nan"))
            ms = data.get("metrics_summary", {})
            if ms:
                print(
                    f"{fname:<26} | {ep:<4d} | {loss:<8.4f} | "
                    f"{ms.get('rec_mel_avg', float('nan')):<7.4f} | "
                    f"{ms.get('rec_oct_avg', float('nan')):<7.4f} | "
                    f"{ms.get('cross_avg', float('nan')):<7.4f} | "
                    f"{ms.get('cos_align_avg', float('nan')):<7.4f} | "
                    f"{ms.get('distill_avg', float('nan')):<7.4f} | "
                    f"{ms.get('vq_loss_avg', float('nan')):<7.4f} | "
                    f"{ms.get('cls_loss_avg', float('nan')):<7.4f} | "
                    f"{ms.get('fm_loss_avg', float('nan')):<7.4f}"
                )
            else:
                print(f"{fname:<26} | {ep:<4d} | {loss:<8.4f} | (metrics_summary non presente)")
        except Exception as e:
            print(f"{fname:<26} | ERRORE LETTURA: {e}")

print("\n" + "=" * 116)
print("📈 2. CRONOLOGIA METRICHE DA LOG (Breakdown fine epoca + diagnostica step L1/Perpl/||g_oct||)")
print("=" * 116)

if not log_file or not os.path.exists(log_file):
    candidates = glob.glob("dual_vae_train_*.out") + glob.glob("slurm-*.out") + glob.glob("*.out") + glob.glob("*.err")
    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        log_file = candidates[0]
        print(f"🔍 Rilevato log più recente: {log_file}")
    else:
        log_file = None

if not log_file or not os.path.exists(log_file):
    print("⚠️ Nessun file .out rilevato automaticamente.")
    print("   Specificare il file esplicito: ./inspect_vae_training_interactive.sh <DIR_CKPT> <FILE_LOG.out> [MAX_EPOCH]")
    sys.exit(0)

# Pattern step intermedi:
# Ep [38]  9.9% | G: 7.342 | [RecM: 0.212 | Cross: 0.406 | Cos: 1.938 | Dist: 3.563 | VQ: 0.006 | Cls: 0.902 | Adv: 0.050] | L1: 11.2% | Perpl: 282.6 | ||g_oct||: 6.23 | LR: 1.9e-04 | 53.4s
step_pattern = re.compile(
    r"Ep\s+\[(\d+)\]\s+([\d\.]+)%\s+\|\s+G:\s+([\d\.]+)\s+\|\s+"
    r"\[RecM:\s+([\d\.]+)\s+\|\s+Cross:\s+([\d\.]+)\s+\|\s+Cos:\s+([\d\.]+)\s+\|\s+Dist:\s+([\d\.]+)\s+\|\s+VQ:\s+([\d\.]+)\s+\|\s+Cls:\s+([\d\.]+)\s+\|\s+Adv:\s+([\-\d\.]+)\]\s+\|\s+"
    r"L1:\s+([\d\.]+)%\s+\|\s+Perpl:\s+([\d\.]+)\s+\|\s+\|\|g_oct\|\|:\s+([\d\.]+)\s+\|\s+LR:\s+([\d\.e\-\+]+)"
)

# Pattern riepilogo fine epoca:
# Epoch 038 terminata in 543.3s (9.05 min). Media G-Loss: 7.34047
#   Breakdown: RecM: 0.2065 | Cross: 0.3989 | Cos: 0.6474 | Dist: 1.8006 | VQ: 0.0056 | Cls: 0.8912 | Adv: 0.5612
epoch_end_pattern = re.compile(r"Epoch\s+(\d+)\s+terminata.*?Media G-Loss:\s+([\d\.]+)")
breakdown_pattern = re.compile(
    r"Breakdown:\s+RecM:\s+([\d\.]+)\s+\|\s+Cross:\s+([\d\.]+)\s+\|\s+Cos:\s+([\d\.]+)\s+\|\s+"
    r"Dist:\s+([\d\.]+)\s+\|\s+VQ:\s+([\d\.]+)\s+\|\s+Cls:\s+([\d\.]+)\s+\|\s+Adv:\s+([\-\d\.]+)"
)

epoch_data = {}
last_finished_epoch = None

def ensure_ep(ep_idx):
    if ep_idx not in epoch_data:
        epoch_data[ep_idx] = {
            "count": 0, "G": 0.0, "RecM": 0.0, "Cross": 0.0,
            "Cos_w": 0.0, "Dist_w": 0.0, "VQ": 0.0, "Cls": 0.0, "Adv_w": 0.0,
            "L1": 0.0, "Perpl": 0.0, "grad": 0.0, "LR": 0.0,
            "breakdown": None
        }
    return epoch_data[ep_idx]

with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m_step = step_pattern.search(line)
        if m_step:
            ep = int(m_step.group(1))
            if max_epoch is not None and ep > max_epoch:
                continue
            pct = float(m_step.group(2))
            # Esclude l'ultimo mini-step parziale al 100.0% (1% del totale) per non falsare la media step
            if pct >= 99.9:
                continue
            s = ensure_ep(ep)
            s["count"] += 1
            s["G"] += float(m_step.group(3))
            s["RecM"] += float(m_step.group(4))
            s["Cross"] += float(m_step.group(5))
            s["Cos_w"] += float(m_step.group(6))
            s["Dist_w"] += float(m_step.group(7))
            s["VQ"] += float(m_step.group(8))
            s["Cls"] += float(m_step.group(9))
            s["Adv_w"] += float(m_step.group(10))
            s["L1"] += float(m_step.group(11))
            s["Perpl"] += float(m_step.group(12))
            s["grad"] += float(m_step.group(13))
            s["LR"] = float(m_step.group(14))
            continue

        m_end = epoch_end_pattern.search(line)
        if m_end:
            ep = int(m_end.group(1))
            if max_epoch is not None and ep > max_epoch:
                last_finished_epoch = None
                continue
            last_finished_epoch = ep
            s = ensure_ep(ep)
            s["G_exact"] = float(m_end.group(2))
            continue

        m_bd = breakdown_pattern.search(line)
        if m_bd and last_finished_epoch is not None:
            s = ensure_ep(last_finished_epoch)
            s["breakdown"] = {
                "RecM": float(m_bd.group(1)),
                "Cross": float(m_bd.group(2)),
                "Cos": float(m_bd.group(3)),
                "Dist": float(m_bd.group(4)),
                "VQ": float(m_bd.group(5)),
                "Cls": float(m_bd.group(6)),
                "Adv": float(m_bd.group(7)),
            }
            last_finished_epoch = None

if not epoch_data:
    print(f"⚠️ Nessuna riga di log parsabile trovata in {log_file}")
else:
    print("Nota: RecM/Cross/Cos/Dist/VQ/Cls/Adv usano i valori esatti non-pesati di Breakdown (se epoca conclusa),")
    print("      altrimenti la media degli step parziali (dove Cos, Dist e Adv sono pesati).")
    print("-" * 124)
    print(f"{'Ep':<4} | {'G-Loss':<7} | {'RecM':<6} | {'Cross':<6} | {'Cos':<6} | {'Dist':<6} | {'VQ':<6} | {'Cls':<6} | {'Adv':<6} | {'L1-Match':<8} | {'Perpl':<6} | {'||g_oct||':<8} | {'LR':<8}")
    print("-" * 124)
    for ep in sorted(epoch_data.keys()):
        s = epoch_data[ep]
        c = max(1, s["count"])
        bd = s["breakdown"]
        g_val = s.get("G_exact", s["G"] / c if s["count"] > 0 else float("nan"))
        recm_v = bd["RecM"] if bd else (s["RecM"] / c)
        cross_v = bd["Cross"] if bd else (s["Cross"] / c)
        cos_v = bd["Cos"] if bd else (s["Cos_w"] / c)
        dist_v = bd["Dist"] if bd else (s["Dist_w"] / c)
        vq_v = bd["VQ"] if bd else (s["VQ"] / c)
        cls_v = bd["Cls"] if bd else (s["Cls"] / c)
        adv_v = bd["Adv"] if bd else (s["Adv_w"] / c)
        l1_v = (s["L1"] / c) if s["count"] > 0 else float("nan")
        perp_v = (s["Perpl"] / c) if s["count"] > 0 else float("nan")
        grad_v = (s["grad"] / c) if s["count"] > 0 else float("nan")
        lr_v = s["LR"]

        print(
            f"{ep:<4d} | {g_val:<7.3f} | {recm_v:<6.3f} | {cross_v:<6.3f} | "
            f"{cos_v:<6.3f} | {dist_v:<6.3f} | {vq_v:<6.3f} | {cls_v:<6.3f} | {adv_v:<6.3f} | "
            f"{l1_v:<7.1f}% | {perp_v:<6.1f} | {grad_v:<8.2f} | {lr_v:<8.1e}"
        )

print("=" * 124)
EOF

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from utils import get_config_from_yaml
from filterbank_diffusion.data.dataset import DistributedAudioRAWDataset
from filterbank_vae.pipeline.spectral import OnlineSpectrogramPipeline
from filterbank_vae.models.vae import MelVQVAE, OctaveVQVAE, DEFAULT_BOTTLENECK_TYPE

LATENT_DIM = 256
NUM_EMBEDDINGS = 1024
NUM_QUANTIZERS = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Latent & Codebook Deep Diagnostics")
    parser.add_argument("--fraction", type=int, default=32, help="Octave fraction (es. 3 o 32)")
    parser.add_argument("--samples_per_class", type=int, default=50, help="Campioni test per classe")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--checkpoint", type=str, default="")
    return parser.parse_args()


def resolve_checkpoint(ckpt_arg):
    if ckpt_arg and os.path.exists(ckpt_arg):
        return ckpt_arg

    env_ckpt = os.environ.get("CHECKPOINT_PATH", "")
    if env_ckpt and os.path.exists(env_ckpt):
        return env_ckpt

    candidate_dirs = [
        os.environ.get("MODEL_CHECKPOINT_DIR", ""),
        os.environ.get("MODELS_GLOBAL", ""),
        os.path.join(src_root, ".models", "vae_model")
    ]
    for d in candidate_dirs:
        if d and os.path.exists(d):
            pts = sorted([f for f in os.listdir(d) if f.startswith("dual_vae_epoch_") and f.endswith(".pt")])
            if pts:
                return os.path.join(d, pts[-1])
    return None


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes_list, _, _, _, sampling_rate, _, _, seed, _, _, _ = get_config_from_yaml("config0.yaml")
    num_classes = len(classes_list)

    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt_path = resolve_checkpoint(args.checkpoint)
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"❌ [CRITICAL] Nessun checkpoint trovato per la diagnostica.", flush=True)
        sys.exit(1)

    print(f"📦 Caricamento checkpoint VAE: {ckpt_path}", flush=True)
    checkpoint = torch.load(ckpt_path, map_location=device)
    active_latent_dim = checkpoint.get("latent_dim", LATENT_DIM)
    active_num_embeddings = checkpoint.get("num_embeddings", NUM_EMBEDDINGS)
    active_num_quantizers = checkpoint.get("num_quantizers", NUM_QUANTIZERS)

    mel_vae = MelVQVAE(
        base_channels=64,
        latent_dim=active_latent_dim,
        num_embeddings=active_num_embeddings,
        num_quantizers=active_num_quantizers
    ).to(device)

    octave_vae = OctaveVQVAE(
        base_channels=64,
        latent_dim=active_latent_dim,
        num_embeddings=active_num_embeddings,
        num_quantizers=active_num_quantizers,
        shared_quantizer=mel_vae.quantizer
    ).to(device)

    mel_state = {k.replace('module.', ''): v for k, v in checkpoint['mel_vae_state_dict'].items()}
    octave_state = {k.replace('module.', ''): v for k, v in checkpoint['octave_vae_state_dict'].items()}
    mel_vae.load_state_dict(mel_state)
    octave_vae.load_state_dict(octave_state)
    mel_vae.eval()
    octave_vae.eval()

    weights_path = os.environ.get("LOCAL_CLAP_WEIGHTS_PATH", ".clap_weights/CLAP_weights_2023.pth")
    spectral_pipeline = OnlineSpectrogramPipeline(weights_path=weights_path, sample_rate=sampling_rate, device=device).to(device)

    raw_dataset_root = os.path.join(os.environ.get("BASEDIR", "/tmp"), "dataSEC", "RAW_DATASET", "raw_wav")
    dataset = DistributedAudioRAWDataset(base_dir=raw_dataset_root, split="test", target_samples_per_class=args.samples_per_class)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    output_dir = os.path.join(src_root, "results", "vae_latent_diagnostics", f"fraction_1_{args.fraction}")
    os.makedirs(output_dir, exist_ok=True)

    # Accumulatori diagnostici
    token_usage_per_class = {c: np.zeros(active_num_embeddings, dtype=np.float64) for c in classes_list}
    class_band_metrics = {c: {"l2": [[] for _ in range(4)], "cos": [[] for _ in range(4)], "tok_acc": [[] for _ in range(4)]} for c in classes_list}
    sample_discrepancy_records = []

    print(f"🚀 Avvio scansione diagnostica su {len(dataset)} campioni (Frazione 1/{args.fraction})...", flush=True)

    with torch.no_grad():
        for b_idx, (raw_audio, class_labels) in enumerate(dataloader):
            raw_audio = raw_audio.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)

            x_mel, x_oct = spectral_pipeline(raw_audio, format_id=1, fraction_id=args.fraction, device=device)

            # Estrazione bottleneck Mel e Octave
            z_e_mel = mel_vae.encoder(x_mel)
            z_q_mel, _, indices_mel = mel_vae.quantizer(z_e_mel, update_ema=False)

            z_e_oct = octave_vae.encoder(x_oct)
            z_q_oct, _, indices_oct = octave_vae.quantizer(z_e_oct, update_ema=False)

            # Dimensioni: [B, D, 4, 72] e [B, num_q, 4, 72]
            l1_indices_mel = indices_mel[:, 0, :, :]  # [B, 4, 72]
            l1_indices_oct = indices_oct[:, 0, :, :]  # [B, 4, 72]

            bs = raw_audio.shape[0]
            for i in range(bs):
                c_idx = class_labels[i].item()
                c_name = classes_list[c_idx]

                # Conteggio occorrenze token (solo ramo octave, che guiderà l'inferenza)
                oct_tokens = l1_indices_oct[i].flatten().cpu().numpy()
                np.add.at(token_usage_per_class[c_name], oct_tokens, 1.0)

                # Profilo per le 4 bande di frequenza latente (H=4)
                band_l2_vals = []
                for h in range(4):
                    z_q_mel_h = z_q_mel[i, :, h, :]  # [D, 72]
                    z_q_oct_h = z_q_oct[i, :, h, :]  # [D, 72]
                    z_e_mel_h = z_e_mel[i, :, h, :]  # [D, 72]
                    z_e_oct_h = z_e_oct[i, :, h, :]  # [D, 72]

                    l2_val = torch.norm(z_q_mel_h - z_q_oct_h, p=2, dim=0).mean().item()
                    cos_val = F.cosine_similarity(z_e_mel_h, z_e_oct_h, dim=0).mean().item()
                    tok_match = (l1_indices_mel[i, h, :] == l1_indices_oct[i, h, :]).float().mean().item() * 100.0

                    class_band_metrics[c_name]["l2"][h].append(l2_val)
                    class_band_metrics[c_name]["cos"][h].append(cos_val)
                    class_band_metrics[c_name]["tok_acc"][h].append(tok_match)
                    band_l2_vals.append(l2_val)

                total_l2 = float(np.mean(band_l2_vals))
                sample_discrepancy_records.append({
                    "sample_idx": b_idx * args.batch_size + i,
                    "class": c_name,
                    "total_l2_dist": total_l2,
                    "l2_band_0_subbass": band_l2_vals[0],
                    "l2_band_1_midlow": band_l2_vals[1],
                    "l2_band_2_midhigh": band_l2_vals[2],
                    "l2_band_3_high": band_l2_vals[3],
                })

    # ------------------------------------------------------------------
    # 1. REPORT BANDE FREQUENZIALI LATENTI
    # ------------------------------------------------------------------
    band_summary_rows = []
    for c_name in classes_list:
        row = {"class": c_name}
        for h, label in enumerate(["0_subbass", "1_midlow", "2_midhigh", "3_high"]):
            row[f"l2_band_{label}"] = float(np.mean(class_band_metrics[c_name]["l2"][h]))
            row[f"cos_band_{label}"] = float(np.mean(class_band_metrics[c_name]["cos"][h]))
            row[f"acc_band_{label}"] = float(np.mean(class_band_metrics[c_name]["tok_acc"][h]))
        band_summary_rows.append(row)

    df_bands = pd.DataFrame(band_summary_rows)
    bands_csv = os.path.join(output_dir, "latent_bands_discrepancy.csv")
    df_bands.to_csv(bands_csv, index=False)
    print(f"📊 Report per bande frequenziali salvato in: {bands_csv}", flush=True)

    # ------------------------------------------------------------------
    # 2. MATRICE DI SOVRAPPOSIZIONE DEI TOKEN (INTERSEZIONE ISTOGRAMMI)
    # ------------------------------------------------------------------
    token_overlap_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    normalized_occupancy = {}
    for c_idx, c_name in enumerate(classes_list):
        tot = token_usage_per_class[c_name].sum()
        normalized_occupancy[c_name] = token_usage_per_class[c_name] / (tot + 1e-10)

    for i, c1 in enumerate(classes_list):
        for j, c2 in enumerate(classes_list):
            p1 = normalized_occupancy[c1]
            p2 = normalized_occupancy[c2]
            intersection = np.sum(np.minimum(p1, p2))
            token_overlap_matrix[i, j] = intersection

    df_overlap = pd.DataFrame(token_overlap_matrix, index=classes_list, columns=classes_list)
    overlap_csv = os.path.join(output_dir, "token_overlap_matrix.csv")
    df_overlap.to_csv(overlap_csv)
    print(f"🔄 Matrice sovrapposizione codebook salvata in: {overlap_csv}", flush=True)

    # ------------------------------------------------------------------
    # 3. IDENTIFICAZIONE DEI TOKEN PIÙ CONTESI (COLLASSO DI SOTTOSPAZIO)
    # ------------------------------------------------------------------
    overall_usage = np.zeros(active_num_embeddings, dtype=np.float64)
    class_usage_count = np.zeros(active_num_embeddings, dtype=np.int32)
    for c_name in classes_list:
        p = normalized_occupancy[c_name]
        overall_usage += p
        class_usage_count += (p > 0.005).astype(np.int32)

    contested_tokens = []
    top_indices = np.argsort(overall_usage)[::-1]
    for tidx in top_indices[:50]:
        if overall_usage[tidx] > 0:
            sharing_classes = [c for c in classes_list if normalized_occupancy[c][tidx] > 0.005]
            contested_tokens.append({
                "token_id": int(tidx),
                "total_usage_score": float(overall_usage[tidx]),
                "num_sharing_classes": len(sharing_classes),
                "classes": sharing_classes
            })

    contested_json = os.path.join(output_dir, "top_contested_tokens.json")
    with open(contested_json, "w") as f:
        json.dump(contested_tokens, f, indent=2)
    print(f"⚠️ Ranking token contesi salvato in: {contested_json}", flush=True)

    # ------------------------------------------------------------------
    # 4. RANKING DEI CAMPIONI PIÙ PROBLEMATICI (HARD SAMPLES)
    # ------------------------------------------------------------------
    df_samples = pd.DataFrame(sample_discrepancy_records)
    df_samples_sorted = df_samples.sort_values(by="total_l2_dist", ascending=False)
    samples_csv = os.path.join(output_dir, "hardest_samples_discrepancy.csv")
    df_samples_sorted.to_csv(samples_csv, index=False)
    print(f"🎯 Ranking hard samples salvato in: {samples_csv}", flush=True)

    dataset.close()
    print("\n🏁 Diagnosi completa dei latenti terminata con successo.\n", flush=True)


if __name__ == "__main__":
    main()

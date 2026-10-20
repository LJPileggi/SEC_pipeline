import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import scipy.stats

current_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from utils import get_config_from_yaml
from filterbank_diffusion.data.dataset import DistributedAudioRAWDataset
from filterbank_vae.pipeline.spectral import OnlineSpectrogramPipeline
from filterbank_vae.models.vae import MelVQVAE, OctaveVQVAE
from filterbank_refiner.models.unet_refiner import MelRefinerUNet

LATENT_DIM = int(os.environ.get("LATENT_DIM", 256))
NUM_EMBEDDINGS = int(os.environ.get("NUM_EMBEDDINGS", 1024))
NUM_QUANTIZERS = int(os.environ.get("NUM_QUANTIZERS", 4))


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Validation: Dual VAE + Residual Refiner")
    parser.add_argument("--vae_checkpoint", type=str, default=os.environ.get("LOCAL_VAE_CHECKPOINT", ""))
    parser.add_argument("--refiner_checkpoint", type=str, default=os.environ.get("LOCAL_REFINER_CHECKPOINT", ""))
    parser.add_argument("--samples_per_class", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def calculate_metrics(p_tensor, q_tensor):
    p_clean = torch.nan_to_num(p_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    q_clean = torch.nan_to_num(q_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    mae = F.l1_loss(p_clean, q_clean).item()
    diff = p_clean - q_clean
    frob = torch.norm(diff, p='fro').item()

    max_p = torch.max(p_clean).item()
    max_q = torch.max(q_clean).item()
    peak_ratio = (max_q / (max_p + 1e-8)) if max_p > 0 else 1.0

    p_energy = torch.exp(p_clean).flatten().double().cpu().numpy()
    q_energy = torch.exp(q_clean).flatten().double().cpu().numpy()

    p_sum, q_sum = np.sum(p_energy), np.sum(q_energy)
    p_energy = p_energy / p_sum if p_sum > 0 else np.ones_like(p_energy) / len(p_energy)
    q_energy = q_energy / q_sum if q_sum > 0 else np.ones_like(q_energy) / len(q_energy)

    eps = 1e-12
    p_energy = np.clip(p_energy, eps, 1.0)
    q_energy = np.clip(q_energy, eps, 1.0)

    kl_div = scipy.stats.entropy(p_energy, q_energy)
    if np.isinf(kl_div) or np.isnan(kl_div):
        kl_div = 0.0

    return float(mae), float(frob), float(peak_ratio), float(kl_div)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes_list, _, _, _, sampling_rate, _, _, seed, _, _, _ = get_config_from_yaml("config0.yaml")

    weights_path = os.environ.get("LOCAL_CLAP_WEIGHTS_PATH", ".clap_weights/CLAP_weights_2023.pth")
    spectral_pipeline = OnlineSpectrogramPipeline(weights_path=weights_path, sample_rate=sampling_rate, device=device).to(device)

    # 1. Caricamento VAE
    if not args.vae_checkpoint or not os.path.exists(args.vae_checkpoint):
        raise FileNotFoundError(f"Checkpoint VAE non trovato: {args.vae_checkpoint}")

    mel_vae = MelVQVAE(base_channels=64, latent_dim=LATENT_DIM, num_embeddings=NUM_EMBEDDINGS, num_quantizers=NUM_QUANTIZERS).to(device)
    octave_vae = OctaveVQVAE(base_channels=64, latent_dim=LATENT_DIM, num_embeddings=NUM_EMBEDDINGS, num_quantizers=NUM_QUANTIZERS, shared_quantizer=mel_vae.quantizer).to(device)

    ckpt_vae = torch.load(args.vae_checkpoint, map_location=device)
    mel_vae.load_state_dict({k.replace('module.', ''): v for k, v in ckpt_vae['mel_vae_state_dict'].items()})
    octave_vae.load_state_dict({k.replace('module.', ''): v for k, v in ckpt_vae['octave_vae_state_dict'].items()})
    mel_vae.eval()
    octave_vae.eval()

    # 2. Caricamento Refiner Diretto
    if not args.refiner_checkpoint or not os.path.exists(args.refiner_checkpoint):
        raise FileNotFoundError(f"Checkpoint Refiner non trovato: {args.refiner_checkpoint}")

    refiner = MelRefinerUNet(in_channels=1, out_channels=1, base_channels=32).to(device)
    ckpt_ref = torch.load(args.refiner_checkpoint, map_location=device)
    ref_state = {k.replace('module.', ''): v for k, v in ckpt_ref['refiner_state_dict'].items()}
    refiner.load_state_dict(ref_state)
    refiner.eval()

    print(f"✅ Modelli caricati con successo:\n   • VAE:     {args.vae_checkpoint}\n   • Refiner: {args.refiner_checkpoint}\n", flush=True)

    raw_dataset_root = os.path.join(os.environ.get("BASEDIR", "/tmp"), "dataSEC", "RAW_DATASET", "raw_wav")
    test_dataset = DistributedAudioRAWDataset(base_dir=raw_dataset_root, split="test", target_samples_per_class=args.samples_per_class)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    output_dir = os.path.join(src_root, "results", "full_pipeline_validation")
    os.makedirs(output_dir, exist_ok=True)

    target_fractions = [1, 3, 6, 12, 16, 24, 32]
    records = []

    for fraction in target_fractions:
        print(f"🔍 Valutazione Pipeline Completa per frazione d'ottava 1/{fraction}...")
        with torch.no_grad():
            for step, (raw_audio, class_labels) in enumerate(test_dataloader):
                raw_audio = raw_audio.to(device, non_blocking=True)
                x_0_pristine, x_oct_pchip = spectral_pipeline(raw_audio, format_id=1, fraction_id=fraction, device=device)

                # Step 1: VAE Cross Reconstruction
                _, _, _, z_q_oct, _ = octave_vae(x_oct_pchip)
                recon_mel_cross_raw = mel_vae.decoder(z_q_oct)

                # Step 2: 1-Step Residual Super-Resolution
                recon_mel_refined = refiner(recon_mel_cross_raw)

                bs = x_0_pristine.shape[0]
                for b in range(bs):
                    mae_raw, _, peak_raw, kl_raw = calculate_metrics(x_0_pristine[b], recon_mel_cross_raw[b])
                    mae_ref, frob_ref, peak_ref, kl_ref = calculate_metrics(x_0_pristine[b], recon_mel_refined[b])

                    c_idx = class_labels[b].item()
                    c_name = classes_list[c_idx] if c_idx < len(classes_list) else "Unknown"

                    records.append({
                        'class': c_name,
                        'octave_fraction': fraction,
                        'mae_cross_raw': mae_raw,
                        'mae_refined': mae_ref,
                        'mae_gain_pct': ((mae_raw - mae_ref) / (mae_raw + 1e-8)) * 100.0,
                        'peak_ratio_raw': peak_raw,
                        'peak_ratio_refined': peak_ref,
                        'frob_refined': frob_ref,
                        'kl_refined': kl_ref
                    })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "detailed_full_pipeline_records.csv"), index=False)

    summary_fraction = df.groupby('octave_fraction')[
        ['mae_cross_raw', 'mae_refined', 'mae_gain_pct', 'peak_ratio_raw', 'peak_ratio_refined', 'frob_refined']
    ].mean().reset_index()

    summary_class = df.groupby('class')[
        ['mae_cross_raw', 'mae_refined', 'mae_gain_pct', 'peak_ratio_raw', 'peak_ratio_refined', 'frob_refined']
    ].mean().reset_index()

    summary_fraction.to_csv(os.path.join(output_dir, "summary_per_fraction.csv"), index=False)
    summary_class.to_csv(os.path.join(output_dir, "summary_per_class.csv"), index=False)

    print("\n" + "=" * 105)
    print("📊 RISULTATI PIPELINE COMPLETA PER FRAZIONE D'OTTAVA (MEDIE)")
    print("=" * 105)
    print(summary_fraction.to_string(index=False))

    print("\n" + "=" * 105)
    print("🎯 RISULTATI PIPELINE COMPLETA PER CLASSE (MEDIE)")
    print("=" * 105)
    print(summary_class.to_string(index=False))
    print("=" * 105)
    print(f"📁 Report salvati in: {output_dir}\n")

    test_dataset.close()


if __name__ == "__main__":
    main()

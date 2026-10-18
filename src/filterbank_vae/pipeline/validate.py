import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import scipy.stats

current_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from utils import get_config_from_yaml
from filterbank_diffusion.data.dataset import DistributedAudioRAWDataset
from filterbank_vae.pipeline.spectral import OnlineSpectrogramPipeline
from filterbank_vae.models.vae import MelVQVAE, OctaveVQVAE

TRAIN_EPOCHS = int(os.environ.get("NUM_EPOCHS", 50))
LATENT_DIM = int(os.environ.get("LATENT_DIM", 256))
NUM_EMBEDDINGS = int(os.environ.get("NUM_EMBEDDINGS", 1024))


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes_list, _, _, _, sampling_rate, _, _, seed, _, _, _ = get_config_from_yaml("config0.yaml")

    samples_per_class = 50
    eval_batch_size = 16
    target_fractions = [1, 3, 6, 12, 16, 24, 32]

    weights_path = os.environ.get("LOCAL_CLAP_WEIGHTS_PATH", ".clap_weights/CLAP_weights_2023.pth")
    spectral_pipeline = OnlineSpectrogramPipeline(weights_path=weights_path, sample_rate=sampling_rate, device=device).to(device)

    mel_vae = MelVQVAE(base_channels=64, latent_dim=LATENT_DIM, num_embeddings=NUM_EMBEDDINGS).to(device)
    octave_vae = OctaveVQVAE(base_channels=64, latent_dim=LATENT_DIM, num_embeddings=NUM_EMBEDDINGS, shared_quantizer=mel_vae.quantizer).to(device)

    target_model_dir = os.environ.get("MODELS_GLOBAL", os.path.join(src_root, ".models", "vae_model"))
    checkpoint_env = os.environ.get("CHECKPOINT_PATH", "")

    if checkpoint_env and os.path.exists(checkpoint_env):
        checkpoint_path = checkpoint_env
    else:
        checkpoint_path = os.path.join(target_model_dir, f"dual_vae_epoch_{TRAIN_EPOCHS - 1}.pt")
        if not os.path.exists(checkpoint_path):
            pts = [f for f in os.listdir(target_model_dir) if f.endswith(".pt")] if os.path.exists(target_model_dir) else []
            if not pts:
                print(f"❌ Nessun checkpoint VQ-GAN trovato in {target_model_dir}. Uscita.", flush=True)
                return
            checkpoint_path = os.path.join(target_model_dir, sorted(pts)[-1])

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    mel_state = {k.replace('module.', ''): v for k, v in checkpoint['mel_vae_state_dict'].items()}
    octave_state = {k.replace('module.', ''): v for k, v in checkpoint['octave_vae_state_dict'].items()}
    
    mel_vae.load_state_dict(mel_state)
    octave_vae.load_state_dict(octave_state)
    mel_vae.eval()
    octave_vae.eval()
    print(f"✅ Checkpoint VQ-GAN caricato con successo: {checkpoint_path}\n", flush=True)

    raw_dataset_root = os.path.join(os.environ.get("BASEDIR", "/tmp"), "dataSEC", "RAW_DATASET", "raw_wav")
    test_dataset = DistributedAudioRAWDataset(base_dir=raw_dataset_root, split="test", target_samples_per_class=samples_per_class)
    test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    output_dir = os.path.join(src_root, "results", "vqvae_diagnostic_validation")
    os.makedirs(output_dir, exist_ok=True)

    records = []

    for fraction in target_fractions:
        print(f"🔍 Valutazione diagnostica per frazione d'ottava 1/{fraction}...")
        
        with torch.no_grad():
            for step, (raw_audio, class_labels) in enumerate(test_dataloader):
                raw_audio = raw_audio.to(device, non_blocking=True)

                x_0_pristine, x_oct_pchip = spectral_pipeline(raw_audio, format_id=1, fraction_id=fraction, device=device)

                # 1. Intra Mel
                recon_mel_intra, _, z_e_mel, z_q_mel, indices_mel = mel_vae(x_0_pristine)

                # 2. Intra Octave
                recon_oct_intra, _, z_e_oct, z_q_oct, indices_oct = octave_vae(x_oct_pchip)

                # 3. Cross Mel (Octave -> Mel)
                recon_mel_cross = mel_vae.decoder(z_q_oct)

                # 4. Statistiche Latenti (B, C, H, W)
                cos_sim_map = F.cosine_similarity(z_e_mel, z_e_oct, dim=1)  # [B, 4, 72]
                l2_dist_map = torch.norm(z_q_mel - z_q_oct, p=2, dim=1)    # [B, 4, 72]

                bs = x_0_pristine.shape[0]
                for b in range(bs):
                    mae_intra_mel, frob_intra_mel, peak_intra_mel, kl_intra_mel = calculate_metrics(x_0_pristine[b], recon_mel_intra[b])
                    mae_intra_oct, frob_intra_oct, peak_intra_oct, kl_intra_oct = calculate_metrics(x_oct_pchip[b], recon_oct_intra[b])
                    mae_cross, frob_cross, peak_cross, kl_cross = calculate_metrics(x_0_pristine[b], recon_mel_cross[b])
                    
                    tok_match = (indices_mel[b] == indices_oct[b]).float().mean().item() * 100.0
                    cos_sim_val = cos_sim_map[b].mean().item()
                    l2_dist_val = l2_dist_map[b].mean().item()

                    c_idx = class_labels[b].item()
                    c_name = classes_list[c_idx] if c_idx < len(classes_list) else "Unknown"

                    records.append({
                        'class': c_name,
                        'octave_fraction': fraction,
                        'token_accuracy_pct': tok_match,
                        'cosine_sim_ze': cos_sim_val,
                        'l2_dist_zq': l2_dist_val,
                        'mae_intra_mel': mae_intra_mel,
                        'mae_intra_octave': mae_intra_oct,
                        'mae_cross_oct': mae_cross,
                        'peak_ratio_intra_mel': peak_intra_mel,
                        'peak_ratio_intra_oct': peak_intra_oct,
                        'peak_ratio_cross': peak_cross,
                        'kl_cross': kl_cross
                    })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "detailed_diagnostic_records.csv"), index=False)

    summary_fraction = df.groupby('octave_fraction')[
        ['token_accuracy_pct', 'cosine_sim_ze', 'l2_dist_zq', 'mae_intra_mel', 'mae_intra_octave', 'mae_cross_oct', 'peak_ratio_cross']
    ].mean().reset_index()

    summary_class = df.groupby('class')[
        ['token_accuracy_pct', 'cosine_sim_ze', 'l2_dist_zq', 'mae_intra_mel', 'mae_intra_octave', 'mae_cross_oct', 'peak_ratio_cross']
    ].mean().reset_index()

    summary_fraction.to_csv(os.path.join(output_dir, "summary_per_fraction.csv"), index=False)
    summary_class.to_csv(os.path.join(output_dir, "summary_per_class.csv"), index=False)

    print("\n" + "=" * 95)
    print("📊 RISULTATI DIAGNOSTICI PER FRAZIONE D'OTTAVA (MEDIE)")
    print("=" * 95)
    print(summary_fraction.to_string(index=False))

    print("\n" + "=" * 95)
    print("🎯 RISULTATI DIAGNOSTICI PER CLASSE (MEDIE)")
    print("=" * 95)
    print(summary_class.to_string(index=False))
    print("=" * 95)
    print(f"📁 Report salvati in: {output_dir}\n")

    test_dataset.close()


if __name__ == "__main__":
    main()

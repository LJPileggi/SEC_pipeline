import os
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from utils import get_config_from_yaml
from filterbank_diffusion.data.dataset import DistributedAudioRAWDataset
from filterbank_vae.pipeline.spectral import OnlineSpectrogramPipeline
from filterbank_vae.models.vae import MelVAE, OctaveVAE, DEFAULT_BOTTLENECK_TYPE

BOTTLENECK_TYPE = os.environ.get("BOTTLENECK_TYPE", DEFAULT_BOTTLENECK_TYPE)
LATENT_DIM = int(os.environ.get("LATENT_DIM", 512))

def plot_triplet(pristine, cond, recon, class_name, sample_idx, fraction, save_path):
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    
    vmin = float(min(pristine.min(), recon.min()))
    vmax = float(max(pristine.max(), recon.max()))
    
    # 1. Pristine Mel [64, 1152]
    im0 = axes[0].imshow(pristine, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Class: {class_name} | #{sample_idx} - Pristine Mel ($X_0$)", fontsize=10, fontweight='bold')
    axes[0].set_ylabel("Mel Bins (64)", fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.015, pad=0.01)
    
    # 2. Input Filterbank PCHIP [320, 1152]
    im1 = axes[1].imshow(cond, origin='lower', aspect='auto', cmap='magma')
    axes[1].set_title(f"Input Degraded Filterbank 1/{fraction} Octave (PCHIP 320 Bands)", fontsize=10)
    axes[1].set_ylabel("Octave Bins (320)", fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.015, pad=0.01)
    
    # 3. Dual-VAE Reconstruction [64, 1152]
    im2 = axes[2].imshow(recon, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
    axes[2].set_title(f"Dual-VAE Restored Mel (Bottleneck: {BOTTLENECK_TYPE.upper()})", fontsize=10)
    axes[2].set_ylabel("Mel Bins (64)", fontsize=8)
    fig.colorbar(im2, ax=axes[2], fraction=0.015, pad=0.01)

    # 4. Errore Assoluto Residuo [64, 1152]
    diff = np.abs(pristine - recon)
    im3 = axes[3].imshow(diff, origin='lower', aspect='auto', cmap='viridis')
    axes[3].set_title(f"Absolute Residual Error $|X_0 - \\hat{{X}}_0|$ (MAE: {diff.mean():.4f})", fontsize=10)
    axes[3].set_xlabel("Time Frames (1152)", fontsize=9)
    axes[3].set_ylabel("Mel Bins (64)", fontsize=8)
    fig.colorbar(im3, ax=axes[3], fraction=0.015, pad=0.01)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=85, format="jpg", pil_kwargs={"quality": 80})
    plt.close()

def resolve_checkpoint_path():
    # 1. Se passata direttamente una variabile esplicita valida
    ckpt_env = os.environ.get("CHECKPOINT_PATH", "")
    if ckpt_env and os.path.exists(ckpt_env):
        return ckpt_env

    # 2. Directory di ricerca in ordine di priorità
    candidate_dirs = [
        os.environ.get("MODEL_CHECKPOINT_DIR", ""),
        os.environ.get("MODELS_GLOBAL", ""),
        os.path.join(src_root, ".models", "vae_model")
    ]

    for d in candidate_dirs:
        if not d or not os.path.exists(d):
            continue
        # Se era specificato un path in CHECKPOINT_PATH ma il path assoluto differiva
        if ckpt_env:
            filename = os.path.basename(ckpt_env)
            candidate_file = os.path.join(d, filename)
            if os.path.exists(candidate_file):
                return candidate_file
        
        # Altrimenti prende l'ultimo per versione
        pts = sorted([f for f in os.listdir(d) if f.startswith("dual_vae_epoch_") and f.endswith(".pt")])
        if pts:
            return os.path.join(d, pts[-1])

    return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes_list, _, _, _, sampling_rate, _, _, _, _, _, _ = get_config_from_yaml("config0.yaml")
    
    samples_per_class = 3
    target_fraction = int(os.environ.get("TARGET_FRACTION", "3"))
    
    weights_path = os.environ.get("LOCAL_CLAP_WEIGHTS_PATH", ".clap_weights/CLAP_weights_2023.pth")
    spectral_pipeline = OnlineSpectrogramPipeline(weights_path=weights_path, sample_rate=sampling_rate, device=device).to(device)
    
    checkpoint_path = resolve_checkpoint_path()
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"❌ Nessun checkpoint Dual-VAE trovato. Verificare percorsi.")
        sys.exit(1)
        
    print(f"✅ Checkpoint caricato: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Allineamento dinamico al bottleneck_type salvato nel checkpoint
    active_bottleneck = checkpoint.get("bottleneck_type", BOTTLENECK_TYPE)
    active_latent_dim = checkpoint.get("latent_dim", LATENT_DIM)
    
    mel_vae = MelVAE(base_channels=64, bottleneck_type=active_bottleneck, latent_dim=active_latent_dim).to(device)
    octave_vae = OctaveVAE(base_channels=64, bottleneck_type=active_bottleneck, latent_dim=active_latent_dim).to(device)
    
    mel_vae.load_state_dict(checkpoint['mel_vae_state_dict'])
    octave_vae.load_state_dict(checkpoint['octave_vae_state_dict'])
    mel_vae.eval()
    octave_vae.eval()
    
    raw_dataset_root = os.path.join(os.environ.get("BASEDIR", "/tmp"), "dataSEC", "RAW_DATASET", "raw_wav")
    test_dataset = DistributedAudioRAWDataset(base_dir=raw_dataset_root, split="test", target_samples_per_class=samples_per_class)
    
    output_dir = os.path.join(src_root, "results", "vae_mel_plots", f"fraction_1_{target_fraction}")
    arrays_dir = os.path.join(output_dir, "arrays_npz")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(arrays_dir, exist_ok=True)
    
    print(f"🎨 Salvataggio immagini JPG in: {output_dir}")
    print(f"💾 Salvataggio array NPZ compressi in: {arrays_dir}")

    class_collected = {c: [] for c in classes_list}
    for i in range(len(test_dataset)):
        raw_audio, class_idx = test_dataset[i]
        c_name = classes_list[class_idx]
        if len(class_collected[c_name]) < samples_per_class:
            class_collected[c_name].append((raw_audio, i))
            
    summary_metrics = {}

    with torch.no_grad():
        for c_name, samples in class_collected.items():
            print(f"   ▶ Generazione campioni per: {c_name}")
            summary_metrics[c_name] = []
            
            for s_idx, (audio_tensor, _) in enumerate(samples):
                audio_batch = audio_tensor.unsqueeze(0).to(device)
                
                x_0_pristine, x_oct_pchip = spectral_pipeline(audio_batch, format_id=1, fraction_id=target_fraction, device=device)
                
                # Cross-Domain Reconstruction: OctaveEncoder -> MelDecoder
                mu_oct, _ = octave_vae.encoder(x_oct_pchip)
                x_reconstructed = mel_vae.decoder(mu_oct)
                
                pristine_np = np.nan_to_num(x_0_pristine.squeeze().cpu().numpy(), nan=0.0).astype(np.float32)
                cond_np = np.nan_to_num(x_oct_pchip.squeeze().cpu().numpy(), nan=0.0).astype(np.float32)
                recon_np = np.nan_to_num(x_reconstructed.squeeze().cpu().numpy(), nan=0.0).astype(np.float32)
                
                base_name = f"{c_name}_sample_{s_idx+1}"
                
                img_path = os.path.join(output_dir, f"{base_name}.jpg")
                plot_triplet(pristine_np, cond_np, recon_np, c_name, s_idx+1, target_fraction, img_path)
                
                npz_path = os.path.join(arrays_dir, f"{base_name}.npz")
                np.savez_compressed(
                    npz_path,
                    pristine=pristine_np,
                    cond=cond_np,
                    recon=recon_np
                )
                
                abs_diff = np.abs(pristine_np - recon_np)
                stat_entry = {
                    "sample": s_idx + 1,
                    "mae": float(abs_diff.mean()),
                    "max_error": float(abs_diff.max()),
                    "pristine_mean": float(pristine_np.mean()),
                    "pristine_std": float(pristine_np.std()),
                    "pristine_min_max": [float(pristine_np.min()), float(pristine_np.max())],
                    "recon_mean": float(recon_np.mean()),
                    "recon_std": float(recon_np.std()),
                    "recon_min_max": [float(recon_np.min()), float(recon_np.max())]
                }
                summary_metrics[c_name].append(stat_entry)

    summary_path = os.path.join(output_dir, "reconstruction_diagnostics.json")
    with open(summary_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)

    test_dataset.close()
    print(f"\n🎉 Tutti i plot salvati in: {output_dir}")
    print(f"📊 Riepilogo statistico salvato in: {summary_path}")

if __name__ == "__main__":
    main()

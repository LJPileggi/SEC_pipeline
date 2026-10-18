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
from filterbank_vae.models.vae import MelVQVAE, OctaveVQVAE, DEFAULT_BOTTLENECK_TYPE

BOTTLENECK_TYPE = os.environ.get("BOTTLENECK_TYPE", DEFAULT_BOTTLENECK_TYPE)
LATENT_DIM = int(os.environ.get("LATENT_DIM", 256))
NUM_EMBEDDINGS = int(os.environ.get("NUM_EMBEDDINGS", 1024))

def plot_quartet(pristine, cond, recon_cross, recon_intra, class_name, sample_idx, fraction, save_path):
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    
    # Range dinamico condiviso per tutti i Mel spettrogrammi
    vmin = float(min(pristine.min(), recon_cross.min(), recon_intra.min()))
    vmax = float(max(pristine.max(), recon_cross.max(), recon_intra.max()))
    
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
    
    # 3. Cross-Domain Restored Mel (Octave -> Mel) [64, 1152]
    im2 = axes[2].imshow(recon_cross, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
    axes[2].set_title("Cross-Domain Restored Mel (OctaveEncoder -> Codebook -> MelDecoder)", fontsize=10)
    axes[2].set_ylabel("Mel Bins (64)", fontsize=8)
    fig.colorbar(im2, ax=axes[2], fraction=0.015, pad=0.01)

    # 4. Pure Mel Autoencoding (Intra-Domain: Mel -> Mel) [64, 1152]
    im3 = axes[3].imshow(recon_intra, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
    axes[3].set_title("Pure Mel Autoencoding (MelEncoder -> Codebook -> MelDecoder)", fontsize=10)
    axes[3].set_xlabel("Time Frames (1152)", fontsize=9)
    axes[3].set_ylabel("Mel Bins (64)", fontsize=8)
    fig.colorbar(im3, ax=axes[3], fraction=0.015, pad=0.01)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=85, format="jpg", pil_kwargs={"quality": 80})
    plt.close()

def resolve_checkpoint_path():
    ckpt_env = os.environ.get("CHECKPOINT_PATH", "")
    if ckpt_env and os.path.exists(ckpt_env):
        return ckpt_env

    candidate_dirs = [
        os.environ.get("MODEL_CHECKPOINT_DIR", ""),
        os.environ.get("MODELS_GLOBAL", ""),
        os.path.join(src_root, ".models", "vae_model")
    ]

    for d in candidate_dirs:
        if not d or not os.path.exists(d):
            continue
        if ckpt_env:
            filename = os.path.basename(ckpt_env)
            candidate_file = os.path.join(d, filename)
            if os.path.exists(candidate_file):
                return candidate_file
        
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
        print("❌ Nessun checkpoint Dual VQ-VAE trovato. Verificare percorsi.")
        sys.exit(1)
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    active_latent_dim = checkpoint.get("latent_dim", LATENT_DIM)
    active_num_embeddings = checkpoint.get("num_embeddings", NUM_EMBEDDINGS)
    
    mel_vae = MelVQVAE(base_channels=64, latent_dim=active_latent_dim, num_embeddings=active_num_embeddings).to(device)
    octave_vae = OctaveVQVAE(base_channels=64, latent_dim=active_latent_dim, num_embeddings=active_num_embeddings, shared_quantizer=mel_vae.quantizer).to(device)
    
    mel_state = {k.replace('module.', ''): v for k, v in checkpoint['mel_vae_state_dict'].items()}
    octave_state = {k.replace('module.', ''): v for k, v in checkpoint['octave_vae_state_dict'].items()}
    
    mel_vae.load_state_dict(mel_state)
    octave_vae.load_state_dict(octave_state)
    mel_vae.eval()
    octave_vae.eval()
    print(f"✅ Checkpoint caricato: {checkpoint_path} (Latent Dim: {active_latent_dim}, Codebook: {active_num_embeddings})")
    
    raw_dataset_root = os.path.join(os.environ.get("BASEDIR", "/tmp"), "dataSEC", "RAW_DATASET", "raw_wav")
    test_dataset = DistributedAudioRAWDataset(base_dir=raw_dataset_root, split="test", target_samples_per_class=samples_per_class)
    
    output_dir = os.path.join(src_root, "results", "vae_mel_plots", f"fraction_1_{target_fraction}")
    arrays_dir = os.path.join(output_dir, "arrays_npz")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(arrays_dir, exist_ok=True)

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
                
                # 1. Ricostruzione pura Mel (Intra-Domain)
                recon_intra, _, _, _, indices_mel = mel_vae(x_0_pristine)
                
                # 2. Traduzione Cross-Domain (Octave -> Mel)
                z_e_oct = octave_vae.encoder(x_oct_pchip)
                z_q_oct, _, indices_oct = octave_vae.quantizer(z_e_oct)
                recon_cross = mel_vae.decoder(z_q_oct)
                
                pristine_np = np.nan_to_num(x_0_pristine.squeeze().cpu().numpy(), nan=0.0).astype(np.float32)
                cond_np = np.nan_to_num(x_oct_pchip.squeeze().cpu().numpy(), nan=0.0).astype(np.float32)
                recon_cross_np = np.nan_to_num(recon_cross.squeeze().cpu().numpy(), nan=0.0).astype(np.float32)
                recon_intra_np = np.nan_to_num(recon_intra.squeeze().cpu().numpy(), nan=0.0).astype(np.float32)
                
                base_name = f"{c_name}_sample_{s_idx+1}"
                
                img_path = os.path.join(output_dir, f"{base_name}.jpg")
                plot_quartet(pristine_np, cond_np, recon_cross_np, recon_intra_np, c_name, s_idx+1, target_fraction, img_path)
                
                npz_path = os.path.join(arrays_dir, f"{base_name}.npz")
                np.savez_compressed(
                    npz_path,
                    pristine=pristine_np,
                    cond=cond_np,
                    recon=recon_cross_np,
                    recon_intra=recon_intra_np
                )
                
                mae_cross = float(np.abs(pristine_np - recon_cross_np).mean())
                mae_intra = float(np.abs(pristine_np - recon_intra_np).mean())
                token_match = float((indices_mel == indices_oct).float().mean().item() * 100.0)
                
                stat_entry = {
                    "sample": s_idx + 1,
                    "mae_cross": mae_cross,
                    "mae_intra": mae_intra,
                    "token_accuracy_pct": token_match,
                    "pristine_peak": float(pristine_np.max()),
                    "recon_cross_peak": float(recon_cross_np.max()),
                    "recon_intra_peak": float(recon_intra_np.max())
                }
                summary_metrics[c_name].append(stat_entry)

    summary_path = os.path.join(output_dir, "reconstruction_diagnostics.json")
    with open(summary_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)

    test_dataset.close()
    print(f"\n🎉 Tutti i plot aggiornati salvati in: {output_dir}")
    print(f"📊 Riepilogo diagnostico salvato in: {summary_path}")

if __name__ == "__main__":
    main()

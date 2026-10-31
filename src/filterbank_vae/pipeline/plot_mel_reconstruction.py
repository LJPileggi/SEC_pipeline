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

def plot_full_diagnostic(pristine_mel, recon_mel_intra, recon_mel_cross, 
                         cond_oct, recon_oct_intra, emb_diff_map,
                         class_name, sample_idx, fraction, tok_acc, save_path):
    fig, axes = plt.subplots(6, 1, figsize=(11, 14), sharex=False)
    
    # Range dinamico Mel
    vmin_mel = float(min(pristine_mel.min(), recon_mel_cross.min(), recon_mel_intra.min()))
    vmax_mel = float(max(pristine_mel.max(), recon_mel_cross.max(), recon_mel_intra.max()))
    
    # Range dinamico Octave
    vmin_oct = float(min(cond_oct.min(), recon_oct_intra.min()))
    vmax_oct = float(max(cond_oct.max(), recon_oct_intra.max()))

    # 1. Pristine Mel
    im0 = axes[0].imshow(pristine_mel, origin='lower', aspect='auto', cmap='magma', vmin=vmin_mel, vmax=vmax_mel)
    axes[0].set_title(f"Class: {class_name} | #{sample_idx} - Pristine Mel ($X_0$)", fontsize=10, fontweight='bold')
    axes[0].set_ylabel("Mel (64)", fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.015, pad=0.01)

    # 2. Pure Mel Autoencoding
    im1 = axes[1].imshow(recon_mel_intra, origin='lower', aspect='auto', cmap='magma', vmin=vmin_mel, vmax=vmax_mel)
    axes[1].set_title("Intra Mel Reconstruction (MelEnc -> Codebook -> MelDec)", fontsize=10)
    axes[1].set_ylabel("Mel (64)", fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.015, pad=0.01)

    # 3. Cross-Domain Restored Mel
    im2 = axes[2].imshow(recon_mel_cross, origin='lower', aspect='auto', cmap='magma', vmin=vmin_mel, vmax=vmax_mel)
    axes[2].set_title("Cross-Domain Restored Mel (OctaveEnc -> Codebook -> MelDec)", fontsize=10)
    axes[2].set_ylabel("Mel (64)", fontsize=8)
    fig.colorbar(im2, ax=axes[2], fraction=0.015, pad=0.01)

    # 4. Input Filterbank PCHIP
    im3 = axes[3].imshow(cond_oct, origin='lower', aspect='auto', cmap='viridis', vmin=vmin_oct, vmax=vmax_oct)
    axes[3].set_title(f"Input Degraded Filterbank 1/{fraction} Octave (PCHIP 320 Bands)", fontsize=10)
    axes[3].set_ylabel("Octave (320)", fontsize=8)
    fig.colorbar(im3, ax=axes[3], fraction=0.015, pad=0.01)

    # 5. Octave Autoencoding
    im4 = axes[4].imshow(recon_oct_intra, origin='lower', aspect='auto', cmap='viridis', vmin=vmin_oct, vmax=vmax_oct)
    axes[4].set_title("Intra Octave Reconstruction (OctaveEnc -> Codebook -> OctaveDec)", fontsize=10)
    axes[4].set_ylabel("Octave (320)", fontsize=8)
    fig.colorbar(im4, ax=axes[4], fraction=0.015, pad=0.01)

    # 6. Mappa Differenza L2 tra Embeddings Quantizzati (4 x 72)
    im5 = axes[5].imshow(emb_diff_map, origin='lower', aspect='auto', cmap='inferno')
    axes[5].set_title(f"Latent Difference $\|z_q^{{mel}} - z_q^{{oct}}\|_2$ (Grid 4x72) | Token Match: {tok_acc:.1f}%", fontsize=10, fontweight='bold')
    axes[5].set_xlabel("Latent Time Steps (72)", fontsize=9)
    axes[5].set_ylabel("Freq Lat (4)", fontsize=8)
    fig.colorbar(im5, ax=axes[5], fraction=0.015, pad=0.01)

    plt.tight_layout()
    plt.savefig(save_path, dpi=90, format="jpg", pil_kwargs={"quality": 85})
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
        print("❌ Nessun checkpoint Dual VQ-VAE trovato.")
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
            print(f"   ▶ Generazione campioni diagnostici per: {c_name}")
            summary_metrics[c_name] = []
            
            for s_idx, (audio_tensor, _) in enumerate(samples):
                audio_batch = audio_tensor.unsqueeze(0).to(device)
                
                x_0_pristine, x_oct_pchip = spectral_pipeline(audio_batch, format_id=1, fraction_id=target_fraction, device=device)
                
                # 1. Mel Intra
                recon_intra, _, z_e_mel, z_q_mel, indices_mel = mel_vae(x_0_pristine)
                
                # 2. Octave Intra
                recon_oct, _, z_e_oct, z_q_oct, indices_oct = octave_vae(x_oct_pchip)

                # 3. Mel Cross
                recon_cross = mel_vae.decoder(z_q_oct)
                
                # Mappa di differenza L2 tra gli embeddings [4, 72]
                l2_diff = torch.norm(z_q_mel - z_q_oct, p=2, dim=1).squeeze(0).cpu().numpy()
                tok_match = float((indices_mel == indices_oct).float().mean().item() * 100.0)

                pristine_np = np.nan_to_num(x_0_pristine.squeeze().cpu().numpy(), nan=0.0).astype(np.float32)
                cond_np = np.nan_to_num(x_oct_pchip.squeeze().cpu().numpy(), nan=0.0).astype(np.float32)
                recon_cross_np = np.nan_to_num(recon_cross.squeeze().cpu().numpy(), nan=0.0).astype(np.float32)
                recon_intra_np = np.nan_to_num(recon_intra.squeeze().cpu().numpy(), nan=0.0).astype(np.float32)
                recon_oct_np = np.nan_to_num(recon_oct.squeeze().cpu().numpy(), nan=0.0).astype(np.float32)
                
                base_name = f"{c_name}_sample_{s_idx+1}"
                img_path = os.path.join(output_dir, f"{base_name}.jpg")
                
                plot_full_diagnostic(
                    pristine_np, recon_intra_np, recon_cross_np,
                    cond_np, recon_oct_np, l2_diff,
                    c_name, s_idx+1, target_fraction, tok_match, img_path
                )
                
                npz_path = os.path.join(arrays_dir, f"{base_name}.npz")
                # np.savez_compressed(
                #     npz_path,
                #     pristine=pristine_np,
                #     cond=cond_np,
                #     recon=recon_cross_np,
                #     recon_intra=recon_intra_np,
                #     recon_oct=recon_oct_np,
                #     l2_diff=l2_diff
                # )
                
                mae_cross = float(np.abs(pristine_np - recon_cross_np).mean())
                mae_intra = float(np.abs(pristine_np - recon_intra_np).mean())
                
                stat_entry = {
                    "sample": s_idx + 1,
                    "mae_cross": mae_cross,
                    "mae_intra": mae_intra,
                    "token_accuracy_pct": tok_match,
                    "l2_latent_diff_mean": float(l2_diff.mean())
                }
                summary_metrics[c_name].append(stat_entry)

    summary_path = os.path.join(output_dir, "reconstruction_diagnostics.json")
    with open(summary_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)

    test_dataset.close()
    print(f"\n🎉 Plot completi a 6 pannelli salvati in: {output_dir}")
    print(f"📊 Riepilogo diagnostico salvato in: {summary_path}")

if __name__ == "__main__":
    main()

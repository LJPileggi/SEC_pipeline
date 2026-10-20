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

from src.utils import get_config_from_yaml
from src.filterbank_diffusion.models.unet import SpectrogramUNet
from src.filterbank_diffusion.models.diffusion import ConditionalGaussianDiffusion
from src.filterbank_diffusion.data.dataset import DistributedAudioRAWDataset
from src.filterbank_diffusion.pipeline.spectral import OnlineSpectrogramPipeline

def plot_triplet(pristine, cond, recon, class_name, sample_idx, fraction, save_path):
    """
    Genera un plot diagnostico a 4 pannelli ottimizzato per peso su disco (<150 KB).
    """
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, sharey=True)
    
    vmin = float(min(pristine.min(), recon.min()))
    vmax = float(max(pristine.max(), recon.max()))
    
    # 1. Pristine Ground Truth
    im0 = axes[0].imshow(pristine, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Class: {class_name} | #{sample_idx} - Pristine Mel ($X_0$)", fontsize=10, fontweight='bold')
    axes[0].set_ylabel("Mel Bins", fontsize=9)
    fig.colorbar(im0, ax=axes[0], fraction=0.015, pad=0.01)
    
    # 2. Degraded Condition
    im1 = axes[1].imshow(cond, origin='lower', aspect='auto', cmap='magma')
    axes[1].set_title(f"Degraded Input (1/{fraction} Octave Filterbank)", fontsize=10)
    axes[1].set_ylabel("Mel Bins", fontsize=9)
    fig.colorbar(im1, ax=axes[1], fraction=0.015, pad=0.01)
    
    # 3. Diffusion Restored
    im2 = axes[2].imshow(recon, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
    axes[2].set_title("DDIM Reconstructed Mel ($\\hat{X}_0$)", fontsize=10)
    axes[2].set_ylabel("Mel Bins", fontsize=9)
    fig.colorbar(im2, ax=axes[2], fraction=0.015, pad=0.01)

    # 4. Errore Assoluto Residuo
    diff = np.abs(pristine - recon)
    im3 = axes[3].imshow(diff, origin='lower', aspect='auto', cmap='viridis')
    axes[3].set_title(f"Absolute Residual $|X_0 - \\hat{{X}}_0|$ (MAE: {diff.mean():.4f})", fontsize=10)
    axes[3].set_xlabel("Time Frames", fontsize=9)
    axes[3].set_ylabel("Mel Bins", fontsize=9)
    fig.colorbar(im3, ax=axes[3], fraction=0.015, pad=0.01)
    
    plt.tight_layout()
    # DPI a 85 e salvataggio JPEG per comprimere drasticamente il peso del file
    plt.savefig(save_path, dpi=85, format="jpg", pil_kwargs={"quality": 80})
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes_list, _, _, _, sampling_rate, _, _, _, _, _, _ = get_config_from_yaml("config0.yaml")
    
    samples_per_class = 3
    ddim_steps = 50
    target_fraction = int(os.environ.get("TARGET_FRACTION", "3"))
    
    weights_path = os.environ.get("LOCAL_CLAP_WEIGHTS_PATH", ".clap_weights/CLAP_weights_2023.pth")
    spectral_pipeline = OnlineSpectrogramPipeline(weights_path=weights_path, sample_rate=sampling_rate, device=device).to(device)
    
    unet = SpectrogramUNet(base_channels=64, emb_dim=256).to(device)
    target_model_dir = os.environ.get("MODELS_GLOBAL", os.path.join(src_root, ".models", "diff_model"))
    
    checkpoint_env = os.environ.get("CHECKPOINT_PATH", "")
    if checkpoint_env and os.path.exists(checkpoint_env):
        checkpoint_path = checkpoint_env
    else:
        pts = sorted([f for f in os.listdir(target_model_dir) if f.endswith(".pt")]) if os.path.exists(target_model_dir) else []
        if not pts:
            print(f"❌ Nessun checkpoint trovato in {target_model_dir}. Uscita.")
            return
        checkpoint_path = os.path.join(target_model_dir, pts[-1])
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.eval()
    print(f"✅ Checkpoint caricato: {checkpoint_path}")
    
    diffusion_scheduler = ConditionalGaussianDiffusion(unet_model=unet, timesteps=1000).to(device)
    
    raw_dataset_root = os.path.join(os.environ.get("BASEDIR", "/tmp"), "dataSEC", "RAW_DATASET", "raw_wav")
    test_dataset = DistributedAudioRAWDataset(base_dir=raw_dataset_root, split="test", target_samples_per_class=samples_per_class)
    
    output_dir = os.path.join(src_root, "results", "mel_plots", f"fraction_1_{target_fraction}")
    arrays_dir = os.path.join(output_dir, "arrays_npz")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(arrays_dir, exist_ok=True)
    
    print(f"🎨 Salvataggio immagini JPG in: {output_dir}")
    print(f"💾 Salvataggio dati numerici NPZ in: {arrays_dir}")

    class_collected = {c: [] for c in classes_list}
    for i in range(len(test_dataset)):
        raw_audio, class_idx = test_dataset[i]
        c_name = classes_list[class_idx]
        if len(class_collected[c_name]) < samples_per_class:
            class_collected[c_name].append((raw_audio, i))
            
    summary_metrics = {}

    with torch.no_grad():
        for c_name, samples in class_collected.items():
            print(f"   ▶ Elaborazione classe: {c_name}")
            summary_metrics[c_name] = []
            
            for s_idx, (audio_tensor, global_id) in enumerate(samples):
                audio_batch = audio_tensor.unsqueeze(0).to(device)
                frac_tensor = torch.full((1,), fill_value=float(target_fraction), device=device)
                
                x_0_pristine, x_cond = spectral_pipeline(audio_batch, format_id=1, fraction_id=target_fraction, device=device)
                x_reconstructed = diffusion_scheduler.sample_ddim(x_cond, fraction_id=frac_tensor, ddim_steps=ddim_steps)
                
                pristine_np = np.nan_to_num(x_0_pristine.squeeze().cpu().numpy(), nan=0.0).astype(np.float32)
                cond_np = np.nan_to_num(x_cond.squeeze().cpu().numpy(), nan=0.0).astype(np.float32)
                recon_np = np.nan_to_num(x_reconstructed.squeeze().cpu().numpy(), nan=0.0).astype(np.float32)
                
                base_name = f"{c_name}_sample_{s_idx+1}"
                
                # 1. Salvataggio immagine leggera JPG (~120 KB)
                img_path = os.path.join(output_dir, f"{base_name}.jpg")
                plot_triplet(pristine_np, cond_np, recon_np, c_name, s_idx+1, target_fraction, img_path)
                
                # 2. Salvataggio array numerici compressi (.npz, ~100-200 KB)
                npz_path = os.path.join(arrays_dir, f"{base_name}.npz")
                np.savez_compressed(
                    npz_path,
                    pristine=pristine_np,
                    cond=cond_np,
                    recon=recon_np
                )
                
                # 3. Metriche diagnostiche testuali (per analisi immediata senza file pesanti)
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

    # Salvataggio riepilogo statistico JSON
    summary_path = os.path.join(output_dir, "reconstruction_diagnostics.json")
    with open(summary_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)

    test_dataset.close()
    print(f"\n🎉 Finito! Report statistico salvato in: {summary_path}")

if __name__ == "__main__":
    main()

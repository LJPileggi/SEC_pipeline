import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from src.utils import get_config_from_yaml
from src.filterbank_diffusion.models.unet import SpectrogramUNet
from src.filterbank_diffusion.models.diffusion import ConditionalGaussianDiffusion
from src.filterbank_diffusion.data.dataset import DistributedAudioRAWDataset
from src.filterbank_diffusion.pipeline.spectral import OnlineSpectrogramPipeline

def plot_triplet(pristine, cond, recon, class_name, sample_idx, fraction, save_path):
    """
    Plots a 4-panel diagnostic figure: Pristine, Degraded Conditioning,
    DDIM Reconstruction, and Absolute Residual Error.
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, sharey=True)
    
    # Valori comuni per scala coerente
    vmin = min(pristine.min(), recon.min())
    vmax = max(pristine.max(), recon.max())
    
    # 1. Pristine Ground Truth
    im0 = axes[0].imshow(pristine, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Class: {class_name} | Sample #{sample_idx} - Original Ground Truth (Pristine Mel)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Mel Bins")
    fig.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.01)
    
    # 2. Degraded Condition (1/fraction octave)
    im1 = axes[1].imshow(cond, origin='lower', aspect='auto', cmap='magma')
    axes[1].set_title(f"Degraded Input ($1/{fraction}$ Octave Filterbank Condition)", fontsize=11)
    axes[1].set_ylabel("Mel Bins")
    fig.colorbar(im1, ax=axes[1], fraction=0.02, pad=0.01)
    
    # 3. Diffusion Restored
    im2 = axes[2].imshow(recon, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
    axes[2].set_title("DDIM Generative Restoration (Reconstructed Mel)", fontsize=11)
    axes[2].set_ylabel("Mel Bins")
    fig.colorbar(im2, ax=axes[2], fraction=0.02, pad=0.01)

    # 4. Absolute Error Residual
    diff = np.abs(pristine - recon)
    im3 = axes[3].imshow(diff, origin='lower', aspect='auto', cmap='viridis')
    axes[3].set_title(f"Absolute Residual Error $|X_0 - \\hat{{X}}_0|$ (MAE: {diff.mean():.4f})", fontsize=11)
    axes[3].set_xlabel("Time Frames")
    axes[3].set_ylabel("Mel Bins")
    fig.colorbar(im3, ax=axes[3], fraction=0.02, pad=0.01)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes_list, _, _, _, sampling_rate, _, _, _, _, _, _ = get_config_from_yaml("config0.yaml")
    
    samples_per_class = 3
    ddim_steps = 50  # 50 step DDIM veloci e nitidi
    target_fraction = int(os.environ.get("TARGET_FRACTION", "3"))  # default 1/3 d'ottava
    
    weights_path = os.environ.get("LOCAL_CLAP_WEIGHTS_PATH", ".clap_weights/CLAP_weights_2023.pth")
    spectral_pipeline = OnlineSpectrogramPipeline(weights_path=weights_path, sample_rate=sampling_rate, device=device).to(device)
    
    unet = SpectrogramUNet(base_channels=64, emb_dim=256).to(device)
    target_model_dir = os.environ.get("MODELS_GLOBAL", os.path.join(src_root, ".models", "diff_model"))
    
    # Recupero checkpoint
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
    print(f"✅ Modello caricato: {checkpoint_path}")
    
    diffusion_scheduler = ConditionalGaussianDiffusion(unet_model=unet, timesteps=1000).to(device)
    
    raw_dataset_root = os.path.join(os.environ.get("BASEDIR", "/tmp"), "dataSEC", "RAW_DATASET", "raw_wav")
    test_dataset = DistributedAudioRAWDataset(base_dir=raw_dataset_root, split="test", target_samples_per_class=samples_per_class)
    
    output_dir = os.path.join(src_root, "results", "mel_plots", f"fraction_1_{target_fraction}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"🎨 Salvataggio plot in: {output_dir}")

    # Raccogliamo i campioni per classe
    class_collected = {c: [] for c in classes_list}
    
    for i in range(len(test_dataset)):
        raw_audio, class_idx = test_dataset[i]
        c_name = classes_list[class_idx]
        if len(class_collected[c_name]) < samples_per_class:
            class_collected[c_name].append((raw_audio, i))
            
    print(f"\n🚀 Inizio generazione plot DDIM ({samples_per_class} campioni per classe, frazione 1/{target_fraction})...")
    
    with torch.no_grad():
        for c_name, samples in class_collected.items():
            print(f"   ▶ Generazione spettrogrammi per: {c_name} ({len(samples)} campioni)")
            for s_idx, (audio_tensor, global_id) in enumerate(samples):
                audio_batch = audio_tensor.unsqueeze(0).to(device) # [1, L]
                frac_tensor = torch.full((1,), fill_value=float(target_fraction), device=device)
                
                # Estrazione spettrogrammi
                x_0_pristine, x_cond = spectral_pipeline(audio_batch, format_id=1, fraction_id=target_fraction, device=device)
                
                # Ricostruzione generativa
                x_reconstructed = diffusion_scheduler.sample_ddim(x_cond, fraction_id=frac_tensor, ddim_steps=ddim_steps)
                
                # Conversione numpy
                pristine_np = x_0_pristine.squeeze().cpu().numpy()
                cond_np = x_cond.squeeze().cpu().numpy()
                recon_np = x_reconstructed.squeeze().cpu().numpy()
                
                # Pulizia nan
                pristine_np = np.nan_to_num(pristine_np, nan=0.0)
                cond_np = np.nan_to_num(cond_np, nan=0.0)
                recon_np = np.nan_to_num(recon_np, nan=0.0)
                
                out_filename = f"{c_name}_sample_{s_idx+1}.png"
                save_file = os.path.join(output_dir, out_filename)
                
                plot_triplet(
                    pristine=pristine_np,
                    cond=cond_np,
                    recon=recon_np,
                    class_name=c_name,
                    sample_idx=s_idx+1,
                    fraction=target_fraction,
                    save_path=save_file
                )

    test_dataset.close()
    print(f"\n🎉 Tutti i plot salvati con successo in: {output_dir}")

if __name__ == "__main__":
    main()

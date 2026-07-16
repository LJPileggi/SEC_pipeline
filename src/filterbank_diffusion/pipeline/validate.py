import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import scipy.stats

current_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if src_root not in sys.path: 
    sys.path.insert(0, src_root)

from src.utils import get_config_from_yaml
from src.filterbank_diffusion.models.unet import ConditionalUNet
from src.filterbank_diffusion.models.diffusion import GaussianDiffusion
from src.filterbank_diffusion.data.dataset import DistributedAudioRAWDataset
from src.filterbank_diffusion.pipeline.spectral import OnlineSpectrogramPipeline

def calculate_distribution_metrics(p_tensor, q_tensor):
    """Computes pure geometric and distributional alignment vectors between log-mels[cite: 9]."""
    frob = torch.norm(p_tensor - q_tensor, p='fro').item()
    
    # 2D Softmax projection to construct continuous dynamic energy densities
    p_prob = F.softmax(p_tensor.flatten(), dim=0).cpu().numpy()
    q_prob = F.softmax(q_tensor.flatten(), dim=0).cpu().numpy()
    
    kl_div = scipy.stats.entropy(p_prob, q_prob)
    if np.isinf(kl_div) or np.isnan(kl_div): 
        kl_div = 0.0
    
    wasserstein = scipy.stats.wasserstein_distance(p_prob, q_prob)
    return frob, kl_div, wasserstein

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes_list, _, _, batch_size, sampling_rate, _, _, seed, _, _, _ = get_config_from_yaml("config0.yaml")
    
    # 1. Initialize spectral pipeline purely for ground-truth and filterbank extraction[cite: 17, 18]
    weights_path = os.environ.get("LOCAL_CLAP_WEIGHTS_PATH", ".clap_weights/CLAP_weights_2023.pth")
    spectral_pipeline = OnlineSpectrogramPipeline(weights_path=weights_path, sample_rate=sampling_rate, device=device).to(device)
    
    # 2. Instantiate and load U-Net weights from the fixed hidden directory[cite: 16, 18]
    unet = ConditionalUNet(num_classes=len(classes_list), base_channels=64, emb_dim=256).to(device)
    target_model_dir = os.path.join(src_root, ".models", "diff_model")
    
    checkpoints = [f for f in os.listdir(target_model_dir) if f.endswith(".pt")]
    if not checkpoints:
        print("❌ No checkpoints found in .models/diff_model/. Exiting.")
        return
    latest_checkpoint = sorted(checkpoints)[-1]
    checkpoint_path = os.path.join(target_model_dir, latest_checkpoint)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    unet.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded generative checkpoint: {checkpoint_path}")
    
    diffusion_scheduler = GaussianDiffusion(unet_model=unet, timesteps=1000).to(device)
    
    # 3. Dedicated Isolated Test Set Dataloader
    raw_dataset_root = os.path.join(os.environ.get("BASEDIR", "/tmp"), "dataSEC", "RAW_DATASET", "raw_wav")
    test_dataset = DistributedAudioRAWDataset(base_dir=raw_dataset_root, split="test", target_samples_per_class=100)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Metrics accumulators
    frob_scores, kl_scores, wass_scores = [], [], []
    granular_mel_residuals = np.zeros(64)
    total_samples = 0
    
    print(f"🔬 Evaluating Mel reconstruction discrepancy on {len(test_dataset)} independent samples...")
    
    with torch.no_grad():
        for step, (raw_audio, class_labels) in enumerate(test_dataloader):
            raw_audio = raw_audio.to(device)
            class_labels = class_labels.to(device)
            
            # Lock configuration variables to compute constant benchmark targets (MP3 at 1/3 octave)
            x_0, conditioning_C = spectral_pipeline(raw_audio, format_id=1, fraction_id=3, device=device)
            
            # Execute 1000 steps reverse diffusion loop via the unconditional null label[cite: 15]
            null_labels = torch.full_like(class_labels, fill_value=len(classes_list))
            x_reconstructed = diffusion_scheduler.sample_loop_cfg(conditioning_C, null_labels, guidance_scale=3.0)
            
            # Extract statistics across batch indices
            for b in range(x_0.shape[0]):
                frob, kl, wass = calculate_distribution_metrics(x_0[b], x_reconstructed[b])
                frob_scores.append(frob)
                kl_scores.append(kl)
                wass_scores.append(wass)
                
                # Compute absolute residual per single Mel Bin[cite: 9]
                abs_residual = torch.abs(x_0[b] - x_reconstructed[b]).squeeze(0) # [64, 700]
                mean_mel_profile = torch.mean(abs_residual, dim=-1).cpu().numpy() # [64]
                granular_mel_residuals += mean_mel_profile
                total_samples += 1
                
    # 4. Consolidate and display validation summary reports
    granular_mel_residuals /= total_samples
    print("\n📊 --- GLOBAL SPECTRAL RESTORATION REPORT ---")
    print(f"  • Avg Frobenius Distance (L2 Geometry): {np.mean(frob_scores):.4f}")
    print(f"  • Avg Kullback-Leibler Divergence:    {np.mean(kl_scores):.4f}")
    print(f"  • Avg Wasserstein (Earth Mover):      {np.mean(wass_scores):.4f}")
    
    # Save statistics arrays for boxplot plotting profiles[cite: 9]
    output_analysis_dir = os.path.join(src_root, "logs", "validation_analysis")
    os.makedirs(output_analysis_dir, exist_ok=True)
    np.save(os.path.join(output_analysis_dir, "granular_mel_residuals.npy"), granular_mel_residuals)
    print(f"💾 Granular profiles saved successfully to: {output_analysis_dir}")

    test_dataset.close()

if __name__ == "__main__":
    main()

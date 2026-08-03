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
import gc

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
    frob = torch.norm(p_tensor - q_tensor, p='fro').item()
    p_prob = F.softmax(p_tensor.flatten(), dim=0).cpu().numpy()
    q_prob = F.softmax(q_tensor.flatten(), dim=0).cpu().numpy()
    
    kl_div = scipy.stats.entropy(p_prob, q_prob)
    if np.isinf(kl_div) or np.isnan(kl_div): 
        kl_div = 0.0
    
    wasserstein = scipy.stats.wasserstein_distance(p_prob, q_prob)
    return frob, kl_div, wasserstein

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes_list, _, epochs, _, sampling_rate, _, _, seed, _, _, _ = get_config_from_yaml("config0.yaml")
    
    samples_per_class = 50
    guidance_scale = 3.0
    ddim_steps = 25
    eval_batch_size = 16
    target_fractions = [1, 3, 6, 12, 16, 24, 32]
    
    weights_path = os.environ.get("LOCAL_CLAP_WEIGHTS_PATH", ".clap_weights/CLAP_weights_2023.pth")
    spectral_pipeline = OnlineSpectrogramPipeline(weights_path=weights_path, sample_rate=sampling_rate, device=device).to(device)
    
    unet = ConditionalUNet(num_classes=len(classes_list), base_channels=64, emb_dim=256).to(device)
    
    target_model_dir = os.environ.get("MODELS_GLOBAL", os.path.join(src_root, ".models", "diff_model"))
    checkpoint_path = os.path.join(target_model_dir, f"unet_epoch_{epochs - 1}.pt")
    
    if not os.path.exists(checkpoint_path):
        pts = [f for f in os.listdir(target_model_dir) if f.endswith(".pt")]
        if not pts:
            print(f"❌ Nessun checkpoint trovato in {target_model_dir}. Uscita.")
            return
        checkpoint_path = os.path.join(target_model_dir, sorted(pts)[-1])
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    unet.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded generative checkpoint: {checkpoint_path}")
    
    diffusion_scheduler = GaussianDiffusion(unet_model=unet, timesteps=1000).to(device)
    
    raw_dataset_root = os.path.join(
        os.environ.get("BASEDIR", "/tmp"),
        "dataSEC", "RAW_DATASET", "raw_wav"
    )
    
    # 🎯 CALCOLO IN BLOCCO: DataLoader globale sull'intero dataset di test scompaginato
    test_dataset = DistributedAudioRAWDataset(
        base_dir=raw_dataset_root, 
        split="test", 
        target_samples_per_class=samples_per_class
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=eval_batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    output_dir = os.path.join(src_root, "results", "diffusion_validation_final")
    os.makedirs(output_dir, exist_ok=True)

    master_records = []

    for fraction in target_fractions:
        print(f"\n⚡ ULTRA-FAST DDIM EVALUATION IN BLOCCO (25 STEPS | BATCH {eval_batch_size}) FOR FRACTION 1/{fraction}")
        
        with torch.no_grad():
            for step, (raw_audio, class_labels) in enumerate(test_dataloader):
                raw_audio = raw_audio.to(device, non_blocking=True)
                class_labels = class_labels.to(device, non_blocking=True)
                
                # Processing del batch direttamente in GPU
                x_0, conditioning_C = spectral_pipeline(raw_audio, format_id=1, fraction_id=fraction, device=device)
                
                # 🚀 VETTORIALIZZAZIONE DDIM SU GPU SULL'INTERO BATCH
                x_reconstructed = diffusion_scheduler.sample_ddim_cfg(
                    conditioning_C, class_labels, ddim_steps=ddim_steps, guidance_scale=guidance_scale
                )
                
                for b in range(x_0.shape[0]):
                    frob, kl, wass = calculate_distribution_metrics(x_0[b], x_reconstructed[b])
                    c_idx = class_labels[b].item()
                    c_name = classes_list[c_idx] if c_idx < len(classes_list) else "Unknown"
                    
                    master_records.append({
                        'track_id': f"step_{step}_b_{b}",
                        'class': c_name,
                        'octave_fraction': fraction,
                        'frobenius': frob,
                        'kl_divergence': kl,
                        'wasserstein': wass
                    })
                    
                del raw_audio, class_labels, x_0, conditioning_C, x_reconstructed
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if master_records:
        df_master = pd.DataFrame(master_records)
        master_csv = os.path.join(output_dir, "consolidated_diffusion_tracks.csv")
        df_master.to_csv(master_csv, index=False)
        
        summary = df_master.groupby('octave_fraction')[['frobenius', 'kl_divergence', 'wasserstein']].agg(['mean', 'std']).reset_index()
        summary_csv = os.path.join(output_dir, "summary_metrics_per_fraction.csv")
        summary.to_csv(summary_csv, index=False)
        
        print("\n" + "="*60)
        print("📊 REPORT FINALE VALIDAZIONE IN BLOCCO (DDIM)")
        print(summary.to_string())
        print("="*60)
        print(f"\n💾 Esportazione completata in: {output_dir}")

    test_dataset.close()

if __name__ == "__main__":
    main()

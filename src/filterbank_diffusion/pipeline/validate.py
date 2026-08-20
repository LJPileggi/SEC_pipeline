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

# --- METRICHE PUNTO A PUNTO E DISTRIBUZIONALI LOCALI ---
def calculate_distribution_metrics(p_tensor, q_tensor):
    p_clean = torch.nan_to_num(p_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    q_clean = torch.nan_to_num(q_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. Distanza di Frobenius
    diff = p_clean - q_clean
    frob = torch.norm(diff, p='fro').item()

    # 2. Densità di energia spettrale (Float64 per stabilità numerica)
    p_energy = torch.exp(p_clean).flatten().double().cpu().numpy()
    q_energy = torch.exp(q_clean).flatten().double().cpu().numpy()

    p_sum = np.sum(p_energy)
    q_sum = np.sum(q_energy)
    
    if p_sum > 0: p_energy /= p_sum
    else: p_energy = np.ones_like(p_energy) / len(p_energy)
        
    if q_sum > 0: q_energy /= q_sum
    else: q_energy = np.ones_like(q_energy) / len(q_energy)

    eps = 1e-12
    p_energy = np.clip(p_energy, eps, 1.0)
    q_energy = np.clip(q_energy, eps, 1.0)

    # 3. KL Divergence & Wasserstein 1D
    kl_div = scipy.stats.entropy(p_energy, q_energy)
    if np.isinf(kl_div) or np.isnan(kl_div): kl_div = 0.0

    wasserstein = scipy.stats.wasserstein_distance(p_energy, q_energy)
    if np.isinf(wasserstein) or np.isnan(wasserstein): wasserstein = 0.0

    return float(frob), float(kl_div), float(wasserstein)


# --- METRICHE NON-LINEARI SUI CENTROIDI (MMD & WASSERSTEIN 2D SINKHORN) ---
def compute_agnostic_mmd(x, y, alphas=[0.1, 1.0, 10.0]):
    if not isinstance(x, torch.Tensor): x = torch.from_numpy(x).float()
    if not isinstance(y, torch.Tensor): y = torch.from_numpy(y).float()
        
    if x.ndim > 2: x = x.squeeze()
    if y.ndim > 2: y = y.squeeze()
        
    x_size = x.size(0)
    y_size = y.size(0)
    
    xx = torch.pow(torch.norm(x, dim=1, keepdim=True), 2)
    yy = torch.pow(torch.norm(y, dim=1, keepdim=True), 2)
    
    dist_xx = xx + xx.t() - 2 * torch.mm(x, x.t())
    dist_yy = yy + yy.t() - 2 * torch.mm(y, y.t())
    dist_xy = xx + yy.t() - 2 * torch.mm(x, y.t())
    
    kernel_xx, kernel_yy, kernel_xy = 0.0, 0.0, 0.0
    for alpha in alphas:
        kernel_xx += torch.exp(-dist_xx / (2 * alpha))
        kernel_yy += torch.exp(-dist_yy / (2 * alpha))
        kernel_xy += torch.exp(-dist_xy / (2 * alpha))
        
    mmd = (kernel_xx.sum() / (x_size * x_size) + 
           kernel_yy.sum() / (y_size * y_size) - 
           2 * kernel_xy.sum() / (x_size * y_size))
           
    return torch.clamp(mmd, min=0.0).item()

def compute_agnostic_wasserstein(p_mat, q_mat, epsilon=0.01, max_iter=100):
    if not isinstance(p_mat, torch.Tensor): p_mat = torch.from_numpy(p_mat).float()
    if not isinstance(q_mat, torch.Tensor): q_mat = torch.from_numpy(q_mat).float()
        
    if p_mat.ndim > 2: p_mat = p_mat.squeeze()
    if q_mat.ndim > 2: q_mat = q_mat.squeeze()

    p_profile = torch.mean(p_mat, dim=1) if p_mat.shape[0] == 64 else torch.mean(p_mat, dim=0)
    q_profile = torch.mean(q_mat, dim=1) if q_mat.shape[0] == 64 else torch.mean(q_mat, dim=0)

    a = F.softmax(p_profile, dim=0).unsqueeze(1) 
    b = F.softmax(q_profile, dim=0).unsqueeze(1) 
    
    dim = a.size(0)
    grid = torch.arange(dim, dtype=torch.float32).unsqueeze(1)
    C = torch.pow(grid - grid.t(), 2)
    C = C / C.max() 
    
    K = torch.exp(-C / epsilon)
    u = torch.ones((dim, 1), dtype=torch.float32) / dim
    
    for _ in range(max_iter):
        v = b / (torch.mm(K.t(), u) + 1e-8)
        u = a / (torch.mm(K, v) + 1e-8)
        
    transport_plan = u * K * v.t()
    return torch.sum(transport_plan * C).item()

def compute_exact_native_interclass_separability(class_native_specs_dict):
    classes = list(class_native_specs_dict.keys())
    if len(classes) < 2:
        return 0.0
        
    centroids = {c: np.mean(specs, axis=0) for c, specs in class_native_specs_dict.items()}
    
    frob_distances = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            c_i = torch.from_numpy(centroids[classes[i]]).float()
            c_j = torch.from_numpy(centroids[classes[j]]).float()
            
            frob_val = torch.norm(c_i - c_j, p='fro').item()
            if frob_val > 1e-5:
                frob_distances.append(frob_val)
                
    if frob_distances:
        return float(min(frob_distances))
    else:
        return 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes_list, _, epochs, _, sampling_rate, _, _, seed, _, _, _ = get_config_from_yaml("config0.yaml")
    
    # ==========================================================
    # 🎯 PARAMETRI VALIDAZIONE HARDCODED
    # ==========================================================
    guidance_scale = 0.0        # 0.0 = Ricostruzione Incondizionata / Agnostica
    samples_per_class = 50
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
    global_nonlinear_records = []

    for fraction in target_fractions:
        print(f"\n⚡ ULTRA-FAST DDIM EVALUATION IN BLOCCO (25 STEPS | BATCH {eval_batch_size}) FOR FRACTION 1/{fraction}")
        
        native_specs_list = []
        reconstructed_specs_list = []
        class_native_specs_dict = {}
        
        with torch.no_grad():
            for step, (raw_audio, class_labels) in enumerate(test_dataloader):
                raw_audio = raw_audio.to(device, non_blocking=True)
                class_labels = class_labels.to(device, non_blocking=True)
                
                x_0, conditioning_C = spectral_pipeline(raw_audio, format_id=1, fraction_id=fraction, device=device)
                
                x_0_clean = torch.nan_to_num(x_0, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Token incondizionato (null label = unet.num_classes)
                null_labels = torch.full_like(class_labels, fill_value=unet.num_classes, device=device)

                x_reconstructed = diffusion_scheduler.sample_ddim_cfg(
                    conditioning_C, null_labels, ddim_steps=ddim_steps, guidance_scale=guidance_scale
                )
                x_rec_clean = torch.nan_to_num(x_reconstructed, nan=0.0, posinf=0.0, neginf=0.0)
                
                for b in range(x_0.shape[0]):
                    frob, kl, wass = calculate_distribution_metrics(x_0_clean[b], x_rec_clean[b])
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
                    
                    spec_nat = x_0_clean[b].squeeze(0).cpu().numpy()
                    spec_rec = x_rec_clean[b].squeeze(0).cpu().numpy()
                    
                    native_specs_list.append(spec_nat)
                    reconstructed_specs_list.append(spec_rec)
                    
                    if c_name not in class_native_specs_dict:
                        class_native_specs_dict[c_name] = []
                    class_native_specs_dict[c_name].append(spec_nat)
                    
                del raw_audio, class_labels, x_0, conditioning_C, x_reconstructed, x_0_clean, x_rec_clean
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if native_specs_list and reconstructed_specs_list:
            global_native_centroid = np.mean(native_specs_list, axis=0)
            global_reconstructed_centroid = np.mean(reconstructed_specs_list, axis=0)

            fraction_mmd = compute_agnostic_mmd(global_native_centroid.T, global_reconstructed_centroid.T)
            fraction_wasserstein_2d = compute_agnostic_wasserstein(global_native_centroid, global_reconstructed_centroid)

            time_steps = global_native_centroid.shape[1]
            half_time = time_steps // 2
            h0_baseline = compute_agnostic_wasserstein(global_native_centroid[:, :half_time], global_native_centroid[:, half_time:(half_time * 2)])

            threshold_separability = compute_exact_native_interclass_separability(class_native_specs_dict)

            print(f"\n🌐 METRICHE GLOBALI NON-LINEARI (FRAZIONE 1/{fraction}):")
            print(f"   • MMD (Native vs Reconstructed):            {fraction_mmd:.6f}")
            print(f"   • 2D Wasserstein (Native vs Reconstructed): {fraction_wasserstein_2d:.6f}")
            print(f"   • H0 Baseline Noise:                       {h0_baseline:.6f}")
            print(f"   • Interclass Separability Threshold (Dist_inter): {threshold_separability:.6f}")

            global_nonlinear_records.append({
                'octave_fraction': fraction,
                'MMD_global_centroids': fraction_mmd,
                'Wasserstein_2D_global_centroids': fraction_wasserstein_2d,
                'H0_Wasserstein_baseline': h0_baseline,
                'Interclass_Separability_Threshold': threshold_separability
            })

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

    if global_nonlinear_records:
        df_nonlinear = pd.DataFrame(global_nonlinear_records)
        nonlinear_csv = os.path.join(output_dir, "global_non_linear_distances.csv")
        df_nonlinear.to_csv(nonlinear_csv, index=False)

        print("\n" + "="*60)
        print("🌐 REPORT METRICHE NON-LINEARI GLOBALI (CENTROIDI)")
        print(df_nonlinear.to_string())
        print("="*60)
        print(f"\n💾 Esportazione completata in: {output_dir}")

    test_dataset.close()

if __name__ == "__main__":
    main()

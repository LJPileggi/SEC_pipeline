import os
import sys
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from utils import setup_environ_vars, setup_distributed_environment, cleanup_distributed_environment, get_config_from_yaml
from filterbank_diffusion.models.unet import ConditionalUNet
from filterbank_diffusion.models.diffusion import GaussianDiffusion
from filterbank_diffusion.data.dataset import DistributedAudioRAWDataset
from filterbank_diffusion.pipeline.spectral import OnlineSpectrogramPipeline, SpectralConvergenceLoss

# ==========================================================
# 🎯 PARAMETRI DI ADDESTRAMENTO HARDCODED (HARD CONFIG)
# ==========================================================
USE_CFG = False             # False = 100% Agnostico/Incondizionato | True = CFG con dropout (15%)
LOSS_TYPE = "hybrid"        # "hybrid" = MSE + Spectral Convergence | "mse" = MSE Pura
TRAIN_EPOCHS = 225          # Numero totale di epoche di addestramento
LOCAL_BATCH_SIZE = 12       # Batch size locale per GPU

def main():
    rank, world_size = setup_environ_vars(slurm=True)
    device = setup_distributed_environment(rank, world_size, slurm=True)
    
    classes_list, patience, _, _, sampling_rate, _, _, seed, _, _, _ = get_config_from_yaml("config0.yaml")
    epochs = TRAIN_EPOCHS

    local_seed = seed + rank
    torch.manual_seed(local_seed)
    np.random.seed(local_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(local_seed)

    weights_path = os.environ.get("LOCAL_CLAP_WEIGHTS_PATH", ".clap_weights/CLAP_weights_2023.pth")
    spectral_pipeline = OnlineSpectrogramPipeline(weights_path=weights_path, sample_rate=sampling_rate, device=device).to(device)

    raw_dataset_root = os.path.join(os.environ.get("BASEDIR", "/tmp"), "dataSEC", "RAW_DATASET", "raw_wav")
    dataset = DistributedAudioRAWDataset(base_dir=raw_dataset_root, target_samples_per_class=500)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
    dataloader = DataLoader(dataset, batch_size=LOCAL_BATCH_SIZE, sampler=sampler, num_workers=8, pin_memory=True, drop_last=True)

    unet = ConditionalUNet(num_classes=len(classes_list), base_channels=64, emb_dim=256).to(device)
    diffusion_scheduler = GaussianDiffusion(unet_model=unet, timesteps=1000).to(device)
    spectral_loss_fn = SpectralConvergenceLoss().to(device)
    
    if torch.cuda.is_available():
        unet = nn.parallel.DistributedDataParallel(unet, device_ids=[rank], output_device=rank)
    else:
        unet = nn.parallel.DistributedDataParallel(unet)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=2e-4, weight_decay=1e-4)

    total_steps = len(dataloader)
    print_freq = max(1, total_steps // 10)

    if rank == 0:
        print(f"🏁 DDP Init Complete | GPUs: {world_size} | USE_CFG={USE_CFG} | LOSS={LOSS_TYPE} | Epoche={epochs}")
        print(f"📊 Steps per Epoch: {total_steps} | Print every {print_freq} steps (10%)")

    for epoch in range(epochs):
        unet.train()
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        running_loss = 0.0
        epoch_start_time = time.time()
        step_start_time = time.time()
        
        for step, (raw_audio, class_labels) in enumerate(dataloader):
            raw_audio = raw_audio.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)
            
            format_id = np.random.choice([0, 1])          
            fraction_id = np.random.choice([1, 3, 6, 12, 16, 24, 32]) 
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                x_0, conditioning_C = spectral_pipeline(raw_audio, format_id, fraction_id, device=device)
                
                t = torch.randint(0, 1000, (x_0.shape[0],), device=device).long()
                noise = torch.randn_like(x_0)
                x_t = diffusion_scheduler.q_sample(x_0, t, noise)
                
                # 🎯 MODALITÀ SELEZIONE CLASSE
                if USE_CFG:
                    mask_cfg = torch.rand(class_labels.shape, device=device) < 0.15
                    cfg_labels = torch.where(mask_cfg, torch.tensor(len(classes_list), device=device), class_labels)
                else:
                    cfg_labels = torch.full_like(class_labels, fill_value=len(classes_list), device=device)
                
                noise_pred = unet(x_t, t, conditioning_C, cfg_labels)
                
                # 🎯 MODALITÀ SELEZIONE LOSS
                loss_mse = nn.functional.mse_loss(noise_pred, noise)
                
                if LOSS_TYPE == "hybrid":
                    sqrt_alpha = torch.sqrt(torch.clamp(diffusion_scheduler.alphas_bar[t].view(-1, 1, 1, 1), min=1e-8))
                    sqrt_one_minus_alpha = torch.sqrt(torch.clamp(1.0 - diffusion_scheduler.alphas_bar[t].view(-1, 1, 1, 1), min=0.0))
                    pred_x0 = (x_t - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
                    
                    loss_spec = spectral_loss_fn(pred_x0, x_0)
                    loss = loss_mse + 0.1 * loss_spec
                else:
                    loss = loss_mse
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()
            
            current_loss = loss.item()
            
            if math.isnan(current_loss) or math.isinf(current_loss):
                if rank == 0:
                    print(f"\n❌ [CRITICAL] Loss turned to NaN/Inf at Epoch {epoch}, Step {step}/{total_steps}!")
                sys.exit(1)

            epoch_loss += current_loss
            running_loss += current_loss
            
            if (step + 1) % print_freq == 0 or (step + 1) == total_steps:
                if rank == 0:
                    elapsed = time.time() - step_start_time
                    percent = ((step + 1) / total_steps) * 100
                    avg_step_loss = running_loss / print_freq
                    print(f" ⏱️  Epoch [{epoch:03d}/{epochs:03d}] | Progress: {percent:5.1f}% ({step+1}/{total_steps} steps) | "
                          f"Step Loss: {current_loss:.6f} | Avg 10% Loss: {avg_step_loss:.6f} | Time: {elapsed:.1f}s", flush=True)
                running_loss = 0.0
                step_start_time = time.time()
            
        if rank == 0:
            avg_loss = epoch_loss / total_steps
            total_epoch_time = time.time() - epoch_start_time
            print(f"📢 Epoch {epoch:03d} Complete in {total_epoch_time/60:.2f} min. Master Average Loss: {avg_loss:.6f}\n")
            
            base_model_dir = os.environ.get("MODEL_CHECKPOINT_DIR", os.path.join(src_root, ".models", "diff_model"))
            target_model_dir = base_model_dir if base_model_dir.startswith("/") else os.path.join(src_root, base_model_dir)
            os.makedirs(target_model_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(target_model_dir, f"unet_epoch_{epoch}.pt")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': unet.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            
            print(f"💾 Checkpoint saved cleanly to: {checkpoint_path}")

    dataset.close()
    cleanup_distributed_environment(rank)

if __name__ == "__main__":
    main()

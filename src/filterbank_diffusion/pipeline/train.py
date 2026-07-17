import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np

# Dynamic root injection to safely import core production modules from src/[cite: 13]
current_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

# Import production components and distributed utilities from core architectures[cite: 13]
from utils import setup_environ_vars, setup_distributed_environment, cleanup_distributed_environment, get_config_from_yaml
from filterbank_diffusion.models.unet import ConditionalUNet
from filterbank_diffusion.models.diffusion import GaussianDiffusion
from filterbank_diffusion.data.dataset import DistributedAudioRAWDataset
from filterbank_diffusion.pipeline.spectral import OnlineSpectrogramPipeline

def main():
    # 1. Initialize distributed architecture handlers[cite: 13]
    rank, world_size = setup_environ_vars(slurm=True)
    device = setup_distributed_environment(rank, world_size, slurm=True)
    
    classes_list, patience, epochs, batch_size, sampling_rate, _, _, seed, _, _, _ = get_config_from_yaml("config0.yaml")
    
    local_seed = seed + rank
    torch.manual_seed(local_seed)
    np.random.seed(local_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(local_seed)

    # 2. Extract path variable to load pure isolated HTS-AT layers[cite: 13]
    weights_path = os.environ.get("LOCAL_CLAP_WEIGHTS_PATH", ".clap_weights/CLAP_weights_2023.pth")
    spectral_pipeline = OnlineSpectrogramPipeline(weights_path=weights_path, sample_rate=sampling_rate, device=device).to(device)

    # 3. Distributed Data loading matching storage stage-in layout[cite: 13]
    raw_dataset_root = os.path.join(os.environ.get("BASEDIR", "/tmp"), "dataSEC", "RAW_DATASET", "raw_wav")
    dataset = DistributedAudioRAWDataset(base_dir=raw_dataset_root, target_samples_per_class=500)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True, drop_last=True)

    # 4. Generative Models Initialization[cite: 13]
    unet = ConditionalUNet(num_classes=len(classes_list), base_channels=64, emb_dim=256).to(device)
    diffusion_scheduler = GaussianDiffusion(unet_model=unet, timesteps=1000).to(device)
    
    if torch.cuda.is_available():
        unet = nn.parallel.DistributedDataParallel(unet, device_ids=[rank], output_device=rank)
    else:
        unet = nn.parallel.DistributedDataParallel(unet)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    if rank == 0:
        print(f"🏁 DDP Initialization Complete. Running on {world_size} GPUs. Local Batch: {batch_size}")

    for epoch in range(epochs):
        unet.train()
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        
        for step, (raw_audio, class_labels) in enumerate(dataloader):
            raw_audio = raw_audio.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)
            
            format_id = np.random.choice([0, 1])          
            fraction_id = np.random.choice([1, 3, 6, 12, 16, 24, 32]) 
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                x_0, conditioning_C = spectral_pipeline(raw_audio, format_id, fraction_id, device=device)
                
                t = torch.randint(0, 1000, (x_0.shape[0],), device=device).long()
                noise = torch.randn_like(x_0)
                x_t = diffusion_scheduler.q_sample(x_0, t, noise)
                
                mask_cfg = torch.rand(class_labels.shape, device=device) < 0.15
                cfg_labels = torch.where(mask_cfg, torch.tensor(len(classes_list), device=device), class_labels)
                
                noise_pred = unet(x_t, t, conditioning_C, cfg_labels)
                loss = nn.functional.mse_loss(noise_pred, noise)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
        if rank == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"📢 Epoch {epoch:03d} Complete. Master Average Loss MSE: {avg_loss:.6f}")
            
            # 🎯 DIRECTORY AND CHECKPOINT RESOLUTION
            # Save state dicts inside the specified hidden directory structure
            target_model_dir = os.path.join(src_root, ".models", "diff_model")
            os.makedirs(target_model_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(target_model_dir, f"unet_epoch_{epoch}.pt")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': unet.module.state_dict(), # Correctly extract original weights from the DDP wrapper
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            
            print(f"💾 Checkpoint saved cleanly to: {checkpoint_path}")

    dataset.close()
    cleanup_distributed_environment(rank)

if __name__ == "__main__":
    main()

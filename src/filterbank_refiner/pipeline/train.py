import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from utils import setup_environ_vars, setup_distributed_environment, cleanup_distributed_environment, get_config_from_yaml
from filterbank_diffusion.data.dataset import DistributedAudioRAWDataset
from filterbank_vae.pipeline.spectral import OnlineSpectrogramPipeline, AcousticStructuralLoss
from filterbank_vae.models.vae import MelVQVAE, OctaveVQVAE
from filterbank_refiner.models.unet_refiner import MelRefinerUNet

DEFAULT_EPOCHS = 20
LOCAL_BATCH_SIZE = 16
LATENT_DIM = 256
NUM_EMBEDDINGS = 1024
NUM_QUANTIZERS = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Conditional Flow Matching Refiner for Cross-Domain Mel")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--vae_checkpoint", type=str, default="")
    return parser.parse_args()


def load_frozen_vae(device, ckpt_path):
    mel_vae = MelVQVAE(base_channels=64, latent_dim=LATENT_DIM, num_embeddings=NUM_EMBEDDINGS, num_quantizers=NUM_QUANTIZERS).to(device)
    octave_vae = OctaveVQVAE(base_channels=64, latent_dim=LATENT_DIM, num_embeddings=NUM_EMBEDDINGS, num_quantizers=NUM_QUANTIZERS, shared_quantizer=mel_vae.quantizer).to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint VAE non trovato in {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    mel_state = {k.replace('module.', ''): v for k, v in checkpoint['mel_vae_state_dict'].items()}
    octave_state = {k.replace('module.', ''): v for k, v in checkpoint['octave_vae_state_dict'].items()}

    mel_vae.load_state_dict(mel_state)
    octave_vae.load_state_dict(octave_state)

    mel_vae.eval()
    octave_vae.eval()
    for p in mel_vae.parameters():
        p.requires_grad = False
    for p in octave_vae.parameters():
        p.requires_grad = False

    return mel_vae, octave_vae


def main():
    args = parse_args()
    rank, world_size = setup_environ_vars(slurm=True)
    device = setup_distributed_environment(rank, world_size, slurm=True)

    classes_list, _, _, _, sampling_rate, _, _, seed, _, _, _ = get_config_from_yaml("config0.yaml")

    local_seed = seed + rank
    torch.manual_seed(local_seed)
    np.random.seed(local_seed)

    weights_path = os.environ.get("LOCAL_CLAP_WEIGHTS_PATH", ".clap_weights/CLAP_weights_2023.pth")
    spectral_pipeline = OnlineSpectrogramPipeline(weights_path=weights_path, sample_rate=sampling_rate, device=device).to(device)

    # Risoluzione checkpoint VAE globale / scratch
    vae_ckpt = args.vae_checkpoint
    if not vae_ckpt:
        model_dir = os.environ.get("MODELS_GLOBAL", os.path.join(src_root, ".models", "vae_model"))
        pts = sorted([f for f in os.listdir(model_dir) if f.startswith("dual_vae_epoch_") and f.endswith(".pt")])
        if not pts:
            raise RuntimeError(f"Nessun checkpoint trovato in {model_dir}")
        vae_ckpt = os.path.join(model_dir, pts[-1])

    if rank == 0:
        print(f"❄️ Caricamento Dual VQ-VAE congelato: {vae_ckpt}", flush=True)
    mel_vae, octave_vae = load_frozen_vae(device, vae_ckpt)

    # Inizializzazione U-Net di raffinamento
    refiner = MelRefinerUNet(in_channels=2, out_channels=1, base_channels=32).to(device)
    refiner = nn.parallel.DistributedDataParallel(refiner, device_ids=[rank], output_device=rank)

    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    acoustic_loss_fn = AcousticStructuralLoss(alpha=0.5, beta=0.25).to(device)

    raw_dataset_root = os.path.join(os.environ.get("BASEDIR", "/tmp"), "dataSEC", "RAW_DATASET", "raw_wav")
    dataset = DistributedAudioRAWDataset(base_dir=raw_dataset_root, split="train", target_samples_per_class=500)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
    dataloader = DataLoader(dataset, batch_size=LOCAL_BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)

    save_dir = os.path.join(src_root, ".models", "refiner_model")
    os.makedirs(save_dir, exist_ok=True)

    total_steps = len(dataloader)
    print_freq = max(1, total_steps // 10)

    if rank == 0:
        print("==========================================================", flush=True)
        print(f"🚀 Training Conditional Flow Refiner (Epochs: {args.epochs})", flush=True)
        print(f"   • U-Net Parametri: {sum(p.numel() for p in refiner.parameters() if p.requires_grad):,}", flush=True)
        print(f"   • Batch Size locale: {LOCAL_BATCH_SIZE} (Totale: {LOCAL_BATCH_SIZE * world_size})", flush=True)
        print("==========================================================\n", flush=True)

    for epoch in range(args.epochs):
        refiner.train()
        sampler.set_epoch(epoch)
        running_loss = 0.0
        epoch_start = time.time()

        for step, (raw_audio, _) in enumerate(dataloader):
            raw_audio = raw_audio.to(device, non_blocking=True)
            fraction_id = np.random.choice([1, 3, 6, 12, 16, 24, 32])

            with torch.no_grad():
                x_0_pristine, x_oct_pchip = spectral_pipeline(raw_audio, format_id=1, fraction_id=fraction_id, device=device)
                # Generazione condizione cross-domain dal VAE congelato
                _, _, _, z_q_oct, _ = octave_vae(x_oct_pchip)
                x_cond = mel_vae.decoder(z_q_oct).detach()

            # Conditional Flow Matching: interpolazione lineare da x_cond a x_0_pristine
            B = x_0_pristine.shape[0]
            t = torch.rand(B, device=device).float()
            t_expand = t[:, None, None, None]

            # x_t = (1 - t) * x_cond + t * x_0
            x_t = (1.0 - t_expand) * x_cond + t_expand * x_0_pristine
            v_target = x_0_pristine - x_cond

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                v_pred = refiner(x_t, x_cond, t)
                loss_flow = F.mse_loss(v_pred, v_target)

                # Penalità acustica opzionale sulla traiettoria integrata a t=1
                x_1_pred = x_cond + v_pred
                loss_struct = acoustic_loss_fn(x_1_pred, x_0_pristine)
                total_loss = loss_flow + 0.1 * loss_struct

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += total_loss.item()

            if (step + 1) % print_freq == 0 or (step + 1) == total_steps:
                avg_l = running_loss / print_freq
                if rank == 0:
                    percent = ((step + 1) / total_steps) * 100
                    print(f"⏱️ Epoch [{epoch:02d}/{args.epochs-1:02d}] | Progress: {percent:5.1f}% | Flow Loss: {avg_l:.5f}", flush=True)
                running_loss = 0.0

        scheduler.step()
        if rank == 0:
            elapsed = time.time() - epoch_start
            print(f"📢 Epoch {epoch:02d} completata in {elapsed:.1f}s ({elapsed/60:.2f} min).\n", flush=True)

            if (epoch + 1) % 5 == 0 or epoch == (args.epochs - 1):
                ckpt_path = os.path.join(save_dir, f"refiner_epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'refiner_state_dict': refiner.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss.item(),
                }, ckpt_path)
                print(f"💾 Checkpoint Refiner salvato in: {ckpt_path}\n", flush=True)

    dataset.close()
    cleanup_distributed_environment(rank)


if __name__ == "__main__":
    main()

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
src_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from utils import setup_environ_vars, setup_distributed_environment, cleanup_distributed_environment, get_config_from_yaml
from filterbank_diffusion.data.dataset import DistributedAudioRAWDataset
from filterbank_vae.pipeline.spectral import OnlineSpectrogramPipeline, AcousticStructuralLoss, feature_matching_loss
from filterbank_vae.models.vae import MelVQVAE, OctaveVQVAE
from filterbank_refiner.models.unet_refiner import MelRefinerUNet, RefinerPatchDiscriminator

DEFAULT_EPOCHS = 25
LOCAL_BATCH_SIZE = 16
LATENT_DIM = 256
NUM_EMBEDDINGS = 1024
NUM_QUANTIZERS = 4
ACCUMULATION_STEPS = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Residual-GAN Refiner for Spectrogram Super-Resolution")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr_g", type=float, default=2e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--adv_weight", type=float, default=0.1)
    parser.add_argument("--fm_weight", type=float, default=2.0)
    parser.add_argument("--vae_checkpoint", type=str, default=os.environ.get("LOCAL_VAE_CHECKPOINT", ""))
    return parser.parse_args()


def load_frozen_vae(device, ckpt_path):
    mel_vae = MelVQVAE(base_channels=64, latent_dim=LATENT_DIM, num_embeddings=NUM_EMBEDDINGS, num_quantizers=NUM_QUANTIZERS).to(device)
    octave_vae = OctaveVQVAE(base_channels=64, latent_dim=LATENT_DIM, num_embeddings=NUM_EMBEDDINGS, num_quantizers=NUM_QUANTIZERS, shared_quantizer=mel_vae.quantizer).to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint VAE non trovato: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    mel_vae.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['mel_vae_state_dict'].items()})
    octave_vae.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['octave_vae_state_dict'].items()})

    mel_vae.eval()
    octave_vae.eval()
    for p in mel_vae.parameters():
        p.requires_grad = False
    for p in octave_vae.parameters():
        p.requires_grad = False

    return mel_vae, octave_vae


def high_frequency_gradient_loss(pred, target):
    """Calcola la consistenza sulle derivate prime temporali e spettrali (Sobel differenziali)."""
    d_t_pred = pred[..., :, 1:] - pred[..., :, :-1]
    d_t_tgt = target[..., :, 1:] - target[..., :, :-1]
    loss_t = F.l1_loss(d_t_pred, d_t_tgt)

    d_f_pred = pred[..., 1:, :] - pred[..., :-1, :]
    d_f_tgt = target[..., 1:, :] - target[..., :-1, :]
    loss_f = F.l1_loss(d_f_pred, d_f_tgt)

    return loss_t + 2.0 * loss_f


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

    if not args.vae_checkpoint or not os.path.exists(args.vae_checkpoint):
        raise RuntimeError(f"❌ Checkpoint VAE errato o assente: {args.vae_checkpoint}")

    if rank == 0:
        print(f"❄️ Caricamento Dual VQ-VAE congelato: {args.vae_checkpoint}", flush=True)
    mel_vae, octave_vae = load_frozen_vae(device, args.vae_checkpoint)

    # 1. Generatore U-Net Diretto
    refiner = MelRefinerUNet(in_channels=1, out_channels=1, base_channels=32).to(device)
    refiner = nn.parallel.DistributedDataParallel(refiner, device_ids=[rank], output_device=rank)

    # 2. Discriminatore PatchGAN
    discriminator = RefinerPatchDiscriminator(in_channels=1, ndf=32).to(device)
    discriminator = nn.parallel.DistributedDataParallel(
        discriminator,
        device_ids=[rank],
        output_device=rank,
        broadcast_buffers=False
    )

    optimizer_g = torch.optim.AdamW(refiner.parameters(), lr=args.lr_g, betas=(0.5, 0.9), weight_decay=1e-4)
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.9), weight_decay=1e-4)

    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=args.epochs, eta_min=1e-6)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=args.epochs, eta_min=1e-6)

    acoustic_loss_fn = AcousticStructuralLoss(alpha=1.0, beta=1.0).to(device)

    raw_dataset_root = os.path.join(os.environ.get("BASEDIR", "/tmp"), "dataSEC", "RAW_DATASET", "raw_wav")
    dataset = DistributedAudioRAWDataset(base_dir=raw_dataset_root, split="train", target_samples_per_class=500)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
    dataloader = DataLoader(dataset, batch_size=LOCAL_BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)

    save_dir = os.environ.get("REFINER_CHECKPOINT_DIR", os.path.join(src_root, ".models", "refiner_model"))
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        print(f"📁 Checkpoint salvati in: {save_dir}", flush=True)

    total_steps = len(dataloader)
    print_freq = max(1, total_steps // 10)

    if rank == 0:
        print("==========================================================", flush=True)
        print(f"🚀 Training Residual-GAN Refiner (1-Step Super-Resolution)", flush=True)
        print(f"   • U-Net Generator Parametri:    {sum(p.numel() for p in refiner.parameters() if p.requires_grad):,}", flush=True)
        print(f"   • Patch Discriminator Parametri: {sum(p.numel() for p in discriminator.parameters() if p.requires_grad):,}", flush=True)
        print(f"   • Adv Weight: {args.adv_weight} | FM Weight: {args.fm_weight}", flush=True)
        print("==========================================================\n", flush=True)

    for epoch in range(args.epochs):
        refiner.train()
        discriminator.train()
        sampler.set_epoch(epoch)
        running_g = 0.0
        running_d = 0.0
        epoch_start = time.time()

        for step, (raw_audio, _) in enumerate(dataloader):
            raw_audio = raw_audio.to(device, non_blocking=True)
            fraction_id = np.random.choice([1, 3, 6, 12, 16, 24, 32])

            with torch.no_grad():
                x_0_pristine, x_oct_pchip = spectral_pipeline(raw_audio, format_id=1, fraction_id=fraction_id, device=device)
                _, _, _, z_q_oct, _ = octave_vae(x_oct_pchip)
                x_cond = mel_vae.decoder(z_q_oct).detach()

            # ------------------------------------------------------
            # 1. PASSO GENERATORE
            # ------------------------------------------------------
            optimizer_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                x_refined = refiner(x_cond)

                l1_loss = F.l1_loss(x_refined, x_0_pristine)
                struct_loss = acoustic_loss_fn(x_refined, x_0_pristine)
                hf_loss = high_frequency_gradient_loss(x_refined, x_0_pristine)

                pred_fake, feats_fake = discriminator(x_refined)
                with torch.no_grad():
                    _, feats_real = discriminator(x_0_pristine)

                fm_loss = feature_matching_loss(feats_real, feats_fake)
                adv_loss = -torch.mean(pred_fake)

                loss_g = l1_loss + struct_loss + 2.0 * hf_loss + (args.fm_weight * fm_loss) + (args.adv_weight * adv_loss)

            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=1.0)
            optimizer_g.step()

            # ------------------------------------------------------
            # 2. PASSO DISCRIMINATORE
            # ------------------------------------------------------
            optimizer_d.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred_real_d, _ = discriminator(x_0_pristine)
                pred_fake_d, _ = discriminator(x_refined.detach())

                loss_d_real = torch.mean(F.relu(1.0 - pred_real_d))
                loss_d_fake = torch.mean(F.relu(1.0 + pred_fake_d))
                loss_d = 0.5 * (loss_d_real + loss_d_fake)

            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_d.step()

            running_g += loss_g.item()
            running_d += loss_d.item()

            if (step + 1) % print_freq == 0 or (step + 1) == total_steps:
                avg_g = running_g / print_freq
                avg_d = running_d / print_freq
                if rank == 0:
                    percent = ((step + 1) / total_steps) * 100
                    print(f"⏱️ Epoch [{epoch:02d}/{args.epochs-1:02d}] | Progress: {percent:5.1f}% | "
                          f"G-Loss: {avg_g:.4f} (Adv: {adv_loss.item():.3f}, HF: {hf_loss.item():.4f}) | "
                          f"D-Loss: {avg_d:.4f}", flush=True)
                running_g = 0.0
                running_d = 0.0

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            elapsed = time.time() - epoch_start
            print(f"📢 Epoch {epoch:02d} completata in {elapsed:.1f}s ({elapsed/60:.2f} min).\n", flush=True)

            if (epoch + 1) % 5 == 0 or epoch == (args.epochs - 1):
                ckpt_path = os.path.join(save_dir, f"refiner_epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'refiner_state_dict': refiner.module.state_dict(),
                    'discriminator_state_dict': discriminator.module.state_dict(),
                    'optimizer_g_state_dict': optimizer_g.state_dict(),
                    'optimizer_d_state_dict': optimizer_d.state_dict(),
                }, ckpt_path)
                print(f"💾 Checkpoint Refiner salvato in: {ckpt_path}\n", flush=True)

    dataset.close()
    cleanup_distributed_environment(rank)


if __name__ == "__main__":
    main()

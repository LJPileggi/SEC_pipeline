import os
import sys
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
from filterbank_vae.models.vae import MelVQVAE, OctaveVQVAE, DEFAULT_BOTTLENECK_TYPE

DEFAULT_EPOCHS = 100
LOCAL_BATCH_SIZE = 16
LATENT_DIM = 256
NUM_EMBEDDINGS = 1024
BOTTLENECK_TYPE = os.environ.get("BOTTLENECK_TYPE", DEFAULT_BOTTLENECK_TYPE)

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Dual VQ-VAE Cross-Domain Contrastive Training")
    parser.add_argument("--start_epoch", type=int, default=int(os.environ.get("START_EPOCH", 0)))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("NUM_EPOCHS", DEFAULT_EPOCHS)))
    parser.add_argument("--bottleneck_type", type=str, default=BOTTLENECK_TYPE, choices=["1d", "2d"])
    parser.add_argument("--latent_dim", type=int, default=LATENT_DIM)
    parser.add_argument("--num_embeddings", type=int, default=NUM_EMBEDDINGS)
    return parser.parse_args()

class InfoNCELoss(nn.Module):
    """
    Contrastive Loss InfoNCE per sincronizzare le feature continue prima della quantizzazione.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_a, z_b, world_size=1):
        z_a = F.normalize(z_a, p=2, dim=-1)
        z_b = F.normalize(z_b, p=2, dim=-1)

        if world_size > 1 and dist.is_initialized():
            gather_a = [torch.zeros_like(z_a) for _ in range(world_size)]
            gather_b = [torch.zeros_like(z_b) for _ in range(world_size)]
            dist.all_gather(gather_a, z_a)
            dist.all_gather(gather_b, z_b)
            all_a = torch.cat(gather_a, dim=0)
            all_b = torch.cat(gather_b, dim=0)
        else:
            all_a = z_a
            all_b = z_b

        sim_ab = torch.matmul(z_a, all_b.T) / self.temperature
        sim_ba = torch.matmul(z_b, all_a.T) / self.temperature

        rank = dist.get_rank() if dist.is_initialized() else 0
        local_bs = z_a.size(0)
        labels = torch.arange(local_bs, device=z_a.device) + (rank * local_bs)

        loss_a = F.cross_entropy(sim_ab, labels)
        loss_b = F.cross_entropy(sim_ba, labels)
        return 0.5 * (loss_a + loss_b)

def main():
    args = parse_args()
    start_epoch = args.start_epoch
    num_epochs = args.epochs
    end_epoch = start_epoch + num_epochs
    latent_dim = args.latent_dim
    num_embeddings = args.num_embeddings

    rank, world_size = setup_environ_vars(slurm=True)
    device = setup_distributed_environment(rank, world_size, slurm=True)

    classes_list, _, _, _, sampling_rate, _, _, seed, _, _, _ = get_config_from_yaml("config0.yaml")

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

    # 1. Istanziazione Dual VQ-VAE con Codebook Condiviso
    mel_vae = MelVQVAE(base_channels=64, latent_dim=latent_dim, num_embeddings=num_embeddings).to(device)
    # L'OctaveVQVAE condivide direttamente lo stesso quantizzatore discreto del Mel
    octave_vae = OctaveVQVAE(base_channels=64, latent_dim=latent_dim, num_embeddings=num_embeddings, shared_quantizer=mel_vae.quantizer).to(device)

    trainable_params = list(mel_vae.parameters()) + list(octave_vae.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=end_epoch, eta_min=1e-6)

    base_model_dir = os.environ.get("MODEL_CHECKPOINT_DIR", os.path.join(src_root, ".models", "vae_model"))
    target_model_dir = base_model_dir if base_model_dir.startswith("/") else os.path.join(src_root, base_model_dir)
    os.makedirs(target_model_dir, exist_ok=True)

    # 2. Resumption Logic
    if start_epoch > 0:
        prev_epoch = start_epoch - 1
        ckpt_candidate = os.path.join(target_model_dir, f"dual_vae_epoch_{prev_epoch}.pt")
        if not os.path.exists(ckpt_candidate) and os.environ.get("MODELS_GLOBAL"):
            ckpt_candidate = os.path.join(os.environ.get("MODELS_GLOBAL"), f"dual_vae_epoch_{prev_epoch}.pt")

        if os.path.exists(ckpt_candidate):
            if rank == 0:
                print(f"🔄 [RESUMPTION] Caricamento checkpoint VQ-VAE: {ckpt_candidate}", flush=True)
            checkpoint = torch.load(ckpt_candidate, map_location=device)
            mel_vae.load_state_dict(checkpoint['mel_vae_state_dict'])
            octave_vae.load_state_dict(checkpoint['octave_vae_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            if rank == 0:
                print(f"❌ [CRITICAL] Checkpoint non trovato: {ckpt_candidate}", flush=True)
            sys.exit(1)

    mel_vae = nn.parallel.DistributedDataParallel(mel_vae, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    octave_vae = nn.parallel.DistributedDataParallel(octave_vae, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    acoustic_loss_fn = AcousticStructuralLoss(alpha=1.0, beta=0.5).to(device)
    contrastive_fn = InfoNCELoss(temperature=0.07).to(device)

    total_steps = len(dataloader)
    print_freq = max(1, total_steps // 10)

    if rank == 0:
        print("==========================================================", flush=True)
        print(f"🏁 Dual VQ-VAE Training Initialized (DDP, GPUs: {world_size})", flush=True)
        print(f"   • Architecture: Discrete Codebook ({num_embeddings} tokens, {latent_dim} dims)", flush=True)
        print("   • Loss: AcousticStructuralLoss (alpha=1.0, beta=0.5) + Commitment Loss + Cross-Translation", flush=True)
        print(f"   • Range epoche: [{start_epoch} -> {end_epoch - 1}] ({num_epochs} epoche)", flush=True)
        print("==========================================================\n", flush=True)

    for epoch in range(start_epoch, end_epoch):
        mel_vae.train()
        octave_vae.train()
        sampler.set_epoch(epoch)

        epoch_loss = 0.0
        running_loss = 0.0
        epoch_start_time = time.time()
        step_start_time = time.time()

        for step, (raw_audio, _) in enumerate(dataloader):
            raw_audio = raw_audio.to(device, non_blocking=True)
            fraction_id = np.random.choice([1, 3, 6, 12, 16, 24, 32])

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Generazione online spettrogrammi
                x_mel_target, x_octave_pchip = spectral_pipeline(raw_audio, format_id=1, fraction_id=fraction_id, device=device)

                # Forward Ramo Mel VQ-VAE
                recon_mel, vq_loss_mel, z_e_mel, z_q_mel, indices_mel = mel_vae(x_mel_target)

                # Forward Ramo Octave VQ-VAE (attraverso lo stesso codebook)
                recon_oct, vq_loss_oct, z_e_oct, z_q_oct, indices_oct = octave_vae(x_octave_pchip)

                # 1. Ricostruzione intra-dominio (Acoustic Structural per Mel, L1 per Octave)
                loss_rec_mel = acoustic_loss_fn(recon_mel, x_mel_target)
                loss_rec_oct = F.l1_loss(recon_oct, x_octave_pchip)

                # 2. VQ Commitment Losses
                loss_vq = vq_loss_mel + vq_loss_oct

                # 3. Contrastive Alignment sulle feature continue pre-quantizzazione (pooled globalmente)
                z_cont_mel = F.adaptive_avg_pool2d(z_e_mel, (1, 1)).flatten(1)
                z_cont_oct = F.adaptive_avg_pool2d(z_e_oct, (1, 1)).flatten(1)
                loss_contrastive = contrastive_fn(z_cont_mel, z_cont_oct, world_size=world_size)

                # 4. Allineamento dei token discreti tra i due rami
                # Forza l'output continuo dell'Octave a combaciare con i vettori discreti del Mel target
                loss_latent_align = F.mse_loss(z_e_oct, z_q_mel.detach())

                # 5. Cross-Domain Translation: decodifica del Mel usando direttamente i tasselli quantizzati dell'Octave
                cross_mel_recon = mel_vae.module.decoder(z_q_oct)
                loss_cross = acoustic_loss_fn(cross_mel_recon, x_mel_target)

                total_loss = (loss_rec_mel + loss_rec_oct) + \
                             loss_vq + \
                             0.5 * loss_latent_align + \
                             0.2 * loss_contrastive + \
                             1.0 * loss_cross

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            current_loss = total_loss.item()
            if math.isnan(current_loss) or math.isinf(current_loss):
                if rank == 0:
                    print(f"\n❌ [CRITICAL] Loss NaN/Inf a Epoca {epoch}, Step {step}/{total_steps}!")
                sys.exit(1)

            epoch_loss += current_loss
            running_loss += current_loss

            if (step + 1) % print_freq == 0 or (step + 1) == total_steps:
                if rank == 0:
                    elapsed = time.time() - step_start_time
                    percent = ((step + 1) / total_steps) * 100
                    avg_step_loss = running_loss / print_freq
                    print(f" ⏱️  Epoch [{epoch:03d}/{end_epoch-1:03d}] | Progress: {percent:5.1f}% | "
                          f"Step Loss: {current_loss:.4f} | Avg 10%: {avg_step_loss:.4f} | "
                          f"Cross-L1: {loss_cross.item():.4f} | VQ: {loss_vq.item():.4f} | Time: {elapsed:.1f}s", flush=True)
                running_loss = 0.0
                step_start_time = time.time()

        scheduler.step()

        if rank == 0:
            avg_loss = epoch_loss / total_steps
            total_epoch_time = time.time() - epoch_start_time
            print(f"📢 Epoch {epoch:03d} terminata in {total_epoch_time:.1f}s ({total_epoch_time/60:.2f} min). Media Loss: {avg_loss:.5f}\n", flush=True)

            if (epoch + 1) % 5 == 0 or epoch == (end_epoch - 1):
                checkpoint_path = os.path.join(target_model_dir, f"dual_vae_epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'mel_vae_state_dict': mel_vae.module.state_dict(),
                    'octave_vae_state_dict': octave_vae.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'bottleneck_type': "vq_vae_2d",
                    'latent_dim': latent_dim,
                    'num_embeddings': num_embeddings,
                }, checkpoint_path)
                print(f"💾 Checkpoint Dual VQ-VAE salvato in: {checkpoint_path}", flush=True)

    dataset.close()
    cleanup_distributed_environment(rank)

if __name__ == "__main__":
    main()

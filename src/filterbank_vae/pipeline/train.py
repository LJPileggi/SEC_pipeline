import os
import sys

# Previene la frammentazione CUDA all'avvio[cite: 50]
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

from utils import setup_environ_vars, setup_distributed_environment, cleanup_distributed_environment, get_config_from_yaml[cite: 50]
from filterbank_diffusion.data.dataset import DistributedAudioRAWDataset[cite: 50]
from filterbank_vae.pipeline.spectral import OnlineSpectrogramPipeline, AcousticStructuralLoss, feature_matching_loss[cite: 50]
from filterbank_vae.models.vae import MelVQVAE, OctaveVQVAE, DiscriminatorWithFeatures, DEFAULT_BOTTLENECK_TYPE[cite: 50]

DEFAULT_EPOCHS = 50[cite: 50]
LOCAL_BATCH_SIZE = 8[cite: 50]
ACCUMULATION_STEPS = 2[cite: 50]
LATENT_DIM = 256[cite: 50]
NUM_EMBEDDINGS = 1024[cite: 50]
NUM_QUANTIZERS = 4
DISC_START_EPOCH = int(os.environ.get("DISC_START_EPOCH", 5))[cite: 50]
DISC_WEIGHT = float(os.environ.get("DISC_WEIGHT", 0.1))[cite: 50]

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Dual RVQ-GAN Cross-Domain Training")
    parser.add_argument("--start_epoch", type=int, default=int(os.environ.get("START_EPOCH", 0)))[cite: 50]
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("NUM_EPOCHS", DEFAULT_EPOCHS)))[cite: 50]
    parser.add_argument("--bottleneck_type", type=str, default=DEFAULT_BOTTLENECK_TYPE)[cite: 50]
    parser.add_argument("--latent_dim", type=int, default=LATENT_DIM)[cite: 50]
    parser.add_argument("--num_embeddings", type=int, default=NUM_EMBEDDINGS)[cite: 50]
    parser.add_argument("--num_quantizers", type=int, default=NUM_QUANTIZERS)
    return parser.parse_args()[cite: 50]

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature[cite: 50]

    def forward(self, z_a, z_b, world_size=1):
        z_a = F.normalize(z_a, p=2, dim=-1)[cite: 50]
        z_b = F.normalize(z_b, p=2, dim=-1)[cite: 50]

        if world_size > 1 and dist.is_initialized():[cite: 50]
            gather_a = [torch.zeros_like(z_a) for _ in range(world_size)][cite: 50]
            gather_b = [torch.zeros_like(z_b) for _ in range(world_size)][cite: 50]
            dist.all_gather(gather_a, z_a)[cite: 50]
            dist.all_gather(gather_b, z_b)[cite: 50]
            all_a = torch.cat(gather_a, dim=0)[cite: 50]
            all_b = torch.cat(gather_b, dim=0)[cite: 50]
        else:
            all_a = z_a[cite: 50]
            all_b = z_b[cite: 50]

        sim_ab = torch.matmul(z_a, all_b.T) / self.temperature[cite: 50]
        sim_ba = torch.matmul(z_b, all_a.T) / self.temperature[cite: 50]

        rank = dist.get_rank() if dist.is_initialized() else 0[cite: 50]
        local_bs = z_a.size(0)[cite: 50]
        labels = torch.arange(local_bs, device=z_a.device) + (rank * local_bs)[cite: 50]

        loss_a = F.cross_entropy(sim_ab, labels)[cite: 50]
        loss_b = F.cross_entropy(sim_ba, labels)[cite: 50]
        return 0.5 * (loss_a + loss_b)[cite: 50]

def compute_cosine_alignment_loss(z_e_oct, z_q_mel):
    """
    Calcola la Cosine Distance punto a punto tra lo spazio latente continuo
    dell'ottava (post Cross-Attention) e i vettori quantizzati del Mel.
    """
    cos_sim = F.cosine_similarity(z_e_oct, z_q_mel, dim=1)  # Shape: [B, H, W]
    return torch.mean(1.0 - cos_sim)

def compute_codebook_diagnostics(indices, num_embeddings):
    """
    Calcola utilizzo e perplessità del codebook sul primo livello di quantizzazione (RVQ stage 0).
    indices: [B, num_quantizers, H, W]
    """
    first_level = indices[:, 0, :, :].flatten()
    unique_count = torch.unique(first_level).numel()[cite: 50]
    pct_usage = (unique_count / num_embeddings) * 100.0[cite: 50]

    encodings = F.one_hot(first_level, num_classes=num_embeddings).float()[cite: 50]
    avg_probs = encodings.mean(dim=0)[cite: 50]
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))).item()[cite: 50]
    return unique_count, pct_usage, perplexity

def main():
    args = parse_args()[cite: 50]
    start_epoch = args.start_epoch[cite: 50]
    num_epochs = args.epochs[cite: 50]
    end_epoch = start_epoch + num_epochs[cite: 50]
    latent_dim = args.latent_dim[cite: 50]
    num_embeddings = args.num_embeddings[cite: 50]
    num_quantizers = args.num_quantizers

    rank, world_size = setup_environ_vars(slurm=True)[cite: 50]
    device = setup_distributed_environment(rank, world_size, slurm=True)[cite: 50]

    classes_list, _, _, _, sampling_rate, _, _, seed, _, _, _ = get_config_from_yaml("config0.yaml")[cite: 50]

    local_seed = seed + rank[cite: 50]
    torch.manual_seed(local_seed)[cite: 50]
    np.random.seed(local_seed)[cite: 50]
    if torch.cuda.is_available():[cite: 50]
        torch.cuda.manual_seed_all(local_seed)[cite: 50]

    weights_path = os.environ.get("LOCAL_CLAP_WEIGHTS_PATH", ".clap_weights/CLAP_weights_2023.pth")[cite: 50]
    spectral_pipeline = OnlineSpectrogramPipeline(weights_path=weights_path, sample_rate=sampling_rate, device=device).to(device)[cite: 50]

    raw_dataset_root = os.path.join(os.environ.get("BASEDIR", "/tmp"), "dataSEC", "RAW_DATASET", "raw_wav")[cite: 50]
    dataset = DistributedAudioRAWDataset(base_dir=raw_dataset_root, target_samples_per_class=500)[cite: 50]

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)[cite: 50]
    dataloader = DataLoader(dataset, batch_size=LOCAL_BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)[cite: 50]

    mel_vae = MelVQVAE(
        base_channels=64,
        latent_dim=latent_dim,
        num_embeddings=num_embeddings,
        num_quantizers=num_quantizers
    ).to(device)

    octave_vae = OctaveVQVAE(
        base_channels=64,
        latent_dim=latent_dim,
        num_embeddings=num_embeddings,
        num_quantizers=num_quantizers,
        shared_quantizer=mel_vae.quantizer
    ).to(device)

    discriminator = DiscriminatorWithFeatures(in_channels=1, ndf=64).to(device)[cite: 50]

    gen_params = list(mel_vae.parameters()) + list(octave_vae.parameters())[cite: 50]
    optimizer_g = torch.optim.AdamW(gen_params, lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-5)[cite: 50]
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=end_epoch, eta_min=1e-6)[cite: 50]

    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-5)[cite: 50]
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=end_epoch, eta_min=1e-6)[cite: 50]

    base_model_dir = os.environ.get("MODEL_CHECKPOINT_DIR", os.path.join(src_root, ".models", "vae_model"))[cite: 50]
    target_model_dir = base_model_dir if base_model_dir.startswith("/") else os.path.join(src_root, base_model_dir)[cite: 50]
    os.makedirs(target_model_dir, exist_ok=True)[cite: 50]

    # Ripristino da checkpoint precedente se start_epoch > 0[cite: 50]
    if start_epoch > 0:[cite: 50]
        prev_epoch = start_epoch - 1[cite: 50]
        ckpt_candidate = os.path.join(target_model_dir, f"dual_vae_epoch_{prev_epoch}.pt")[cite: 50]
        if not os.path.exists(ckpt_candidate) and os.environ.get("MODELS_GLOBAL"):[cite: 50]
            ckpt_candidate = os.path.join(os.environ.get("MODELS_GLOBAL"), f"dual_vae_epoch_{prev_epoch}.pt")[cite: 50]

        if os.path.exists(ckpt_candidate):[cite: 50]
            if rank == 0:[cite: 50]
                print(f"🔄 [RESUMPTION] Caricamento checkpoint Dual RVQ-GAN: {ckpt_candidate}", flush=True)
            checkpoint = torch.load(ckpt_candidate, map_location=device)[cite: 50]
            mel_vae.load_state_dict(checkpoint['mel_vae_state_dict'])[cite: 50]
            octave_vae.load_state_dict(checkpoint['octave_vae_state_dict'])[cite: 50]
            if 'discriminator_state_dict' in checkpoint:[cite: 50]
                discriminator.load_state_dict(checkpoint['discriminator_state_dict'])[cite: 50]
            optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])[cite: 50]
            scheduler_g.load_state_dict(checkpoint['scheduler_state_dict'])[cite: 50]
        else:
            if rank == 0:[cite: 50]
                print(f"❌ [CRITICAL] Checkpoint non trovato: {ckpt_candidate}", flush=True)[cite: 50]
            sys.exit(1)[cite: 50]

    mel_vae = nn.parallel.DistributedDataParallel(mel_vae, device_ids=[rank], output_device=rank, find_unused_parameters=False)[cite: 50]
    octave_vae = nn.parallel.DistributedDataParallel(octave_vae, device_ids=[rank], output_device=rank, find_unused_parameters=False)[cite: 50]
    discriminator = nn.parallel.DistributedDataParallel([cite: 50]
        discriminator,[cite: 50]
        device_ids=[rank],[cite: 50]
        output_device=rank,[cite: 50]
        find_unused_parameters=True,[cite: 50]
        broadcast_buffers=False[cite: 50]
    )[cite: 50]

    acoustic_loss_fn = AcousticStructuralLoss(alpha=1.0, beta=0.5).to(device)[cite: 50]
    contrastive_fn = InfoNCELoss(temperature=0.07).to(device)[cite: 50]

    total_steps = len(dataloader)[cite: 50]
    print_freq = max(1, total_steps // 10)[cite: 50]

    if rank == 0:[cite: 50]
        print("==========================================================", flush=True)[cite: 50]
        print(f"🏁 Dual RVQ-GAN Training Initialized (DDP, GPUs: {world_size})", flush=True)
        print(f"   • RVQ Stages: {num_quantizers} | Embeddings per stage: {num_embeddings}", flush=True)
        print(f"   • Batch: {LOCAL_BATCH_SIZE} (x{ACCUMULATION_STEPS} accum = {LOCAL_BATCH_SIZE * ACCUMULATION_STEPS} effettivo)", flush=True)
        print(f"   • Latent Alignment: Cosine Distance su griglia [4, 72]", flush=True)
        print(f"   • Curriculum Learning: Fase 1 [6,12,16] -> Fase 2 [3..24] -> Fase 3 [1..32]", flush=True)
        print(f"   • Adversarial Warm-up: Inizio Discriminatore da Epoca {DISC_START_EPOCH}", flush=True)[cite: 50]
        print(f"   • Range epoche: [{start_epoch} -> {end_epoch - 1}] ({num_epochs} epoche)", flush=True)[cite: 50]
        print("==========================================================\n", flush=True)[cite: 50]

    optimizer_g.zero_grad(set_to_none=True)[cite: 50]
    optimizer_d.zero_grad(set_to_none=True)[cite: 50]

    for epoch in range(start_epoch, end_epoch):[cite: 50]
        mel_vae.train()[cite: 50]
        octave_vae.train()[cite: 50]
        discriminator.train()[cite: 50]
        sampler.set_epoch(epoch)[cite: 50]

        epoch_loss_g = 0.0[cite: 50]
        running_loss_g = 0.0[cite: 50]
        epoch_start_time = time.time()[cite: 50]
        step_start_time = time.time()[cite: 50]

        use_discriminator = (epoch >= DISC_START_EPOCH)[cite: 50]

        for step, (raw_audio, _) in enumerate(dataloader):[cite: 50]
            raw_audio = raw_audio.to(device, non_blocking=True)[cite: 50]

            # ------------------------------------------------------------------
            # 🎯 CURRICULUM LEARNING SULLE FRAZIONI D'OTTAVA
            # ------------------------------------------------------------------
            if epoch < 10:
                # Fase 1: risoluzioni intermedie e geometricamente stabili
                fraction_id = np.random.choice([6, 12, 16])
            elif epoch < 25:
                # Fase 2: espansione graduale
                fraction_id = np.random.choice([3, 6, 12, 16, 24])
            else:
                # Fase 3: spettro completo incluse le frazioni estremali (1/1 e 1/32)
                fraction_id = np.random.choice([1, 3, 6, 12, 16, 24, 32])

            # Generazione spettrogrammi (Float32 nativo su torchaudio)[cite: 50]
            x_mel_target, x_octave_pchip = spectral_pipeline(raw_audio, format_id=1, fraction_id=fraction_id, device=device)[cite: 50]

            # ------------------------------------------------------------------
            # 1. PASSO GENERATORE (Sotto autocast bfloat16 per i modelli neurali)[cite: 50]
            # ------------------------------------------------------------------
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):[cite: 50]
                # 1a. Ramo Mel
                recon_mel, vq_loss_mel, z_e_mel, z_q_mel, indices_mel = mel_vae(x_mel_target)[cite: 50]

                # 1b. Ramo Ottava guidato via Cross-Attention dalle feature Mel continue
                recon_oct, vq_loss_oct, z_e_oct, z_q_oct, indices_oct = octave_vae(x_octave_pchip, z_mel_ref=z_e_mel.detach())

                loss_rec_mel = acoustic_loss_fn(recon_mel, x_mel_target)[cite: 50]
                loss_rec_oct = F.l1_loss(recon_oct, x_octave_pchip)[cite: 50]
                loss_vq = vq_loss_mel + vq_loss_oct[cite: 50]

                # Allineamento contrastivo globale[cite: 50]
                z_cont_mel = F.adaptive_avg_pool2d(z_e_mel, (1, 1)).flatten(1)[cite: 50]
                z_cont_oct = F.adaptive_avg_pool2d(z_e_oct, (1, 1)).flatten(1)[cite: 50]
                loss_contrastive = contrastive_fn(z_cont_mel, z_cont_oct, world_size=world_size)[cite: 50]

                # Allineamento locale punto a punto via Cosine Distance
                loss_latent_align = compute_cosine_alignment_loss(z_e_oct, z_q_mel.detach())

                # Ricostruzione incrociata Mel usando i codici quantizzati del ramo ottava
                cross_mel_recon = mel_vae.module.decoder(z_q_oct)[cite: 50]
                loss_cross = acoustic_loss_fn(cross_mel_recon, x_mel_target)[cite: 50]

                loss_g_base = (loss_rec_mel + loss_rec_oct) + \
                              loss_vq + \
                              0.5 * loss_latent_align + \
                              0.2 * loss_contrastive + \
                              1.0 * loss_cross[cite: 50]

                loss_adv_total = torch.tensor(0.0, device=device)[cite: 50]
                loss_fm_total = torch.tensor(0.0, device=device)[cite: 50]

                if use_discriminator:[cite: 50]
                    pred_fake_mel, feats_fake_mel = discriminator(recon_mel)[cite: 50]
                    pred_fake_cross, feats_fake_cross = discriminator(cross_mel_recon)[cite: 50]

                    with torch.no_grad():[cite: 50]
                        _, feats_real_mel = discriminator(x_mel_target)[cite: 50]

                    loss_fm_mel = feature_matching_loss(feats_real_mel, feats_fake_mel)[cite: 50]
                    loss_fm_cross = feature_matching_loss(feats_real_mel, feats_fake_cross)[cite: 50]
                    loss_fm_total = 0.5 * (loss_fm_mel + loss_fm_cross)[cite: 50]

                    loss_adv_mel = -torch.mean(pred_fake_mel)[cite: 50]
                    loss_adv_cross = -torch.mean(pred_fake_cross)[cite: 50]
                    loss_adv_total = 0.5 * (loss_adv_mel + loss_adv_cross)[cite: 50]

                    loss_g_total = loss_g_base + (0.5 * loss_fm_total) + (DISC_WEIGHT * loss_adv_total)[cite: 50]
                else:
                    loss_g_total = loss_g_base[cite: 50]

                loss_g_scaled = loss_g_total / ACCUMULATION_STEPS[cite: 50]

            loss_g_scaled.backward()[cite: 50]

            # ------------------------------------------------------------------
            # 2. PASSO DISCRIMINATORE[cite: 50]
            # ------------------------------------------------------------------
            loss_d_total = torch.tensor(0.0, device=device)[cite: 50]
            if use_discriminator:[cite: 50]
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):[cite: 50]
                    pred_real, _ = discriminator(x_mel_target)[cite: 50]
                    pred_fake_mel, _ = discriminator(recon_mel.detach())[cite: 50]
                    pred_fake_cross, _ = discriminator(cross_mel_recon.detach())[cite: 50]

                    loss_d_real = torch.mean(F.relu(1.0 - pred_real))[cite: 50]
                    loss_d_fake = 0.5 * (torch.mean(F.relu(1.0 + pred_fake_mel)) + torch.mean(F.relu(1.0 + pred_fake_cross)))[cite: 50]
                    loss_d_total = 0.5 * (loss_d_real + loss_d_fake)[cite: 50]
                    loss_d_scaled = loss_d_total / ACCUMULATION_STEPS[cite: 50]

                loss_d_scaled.backward()[cite: 50]

            # Aggiornamento parametri con accumulo gradienti[cite: 50]
            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == total_steps:[cite: 50]
                torch.nn.utils.clip_grad_norm_(gen_params, max_norm=1.0)[cite: 50]
                optimizer_g.step()[cite: 50]
                optimizer_g.zero_grad(set_to_none=True)[cite: 50]

                if use_discriminator:[cite: 50]
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)[cite: 50]
                    optimizer_d.step()[cite: 50]
                    optimizer_d.zero_grad(set_to_none=True)[cite: 50]

            current_loss_g = loss_g_total.item()[cite: 50]
            if math.isnan(current_loss_g) or math.isinf(current_loss_g):[cite: 50]
                if rank == 0:[cite: 50]
                    print(f"\n❌ [CRITICAL] Loss NaN/Inf a Epoca {epoch}, Step {step}/{total_steps}!")[cite: 50]
                sys.exit(1)[cite: 50]

            epoch_loss_g += current_loss_g[cite: 50]
            running_loss_g += current_loss_g[cite: 50]

            if (step + 1) % print_freq == 0 or (step + 1) == total_steps:[cite: 50]
                unique_c, pct_u, perp = compute_codebook_diagnostics(indices_mel, num_embeddings)
                token_match_l1 = (indices_mel[:, 0] == indices_oct[:, 0]).float().mean().item() * 100.0

                if rank == 0:[cite: 50]
                    elapsed = time.time() - step_start_time[cite: 50]
                    percent = ((step + 1) / total_steps) * 100[cite: 50]
                    avg_g = running_loss_g / print_freq[cite: 50]
                    print(f" ⏱️  Epoch [{epoch:03d}/{end_epoch-1:03d}] | Progress: {percent:5.1f}% | "
                          f"G-Loss: {avg_g:.4f} | D-Loss: {loss_d_total.item():.4f} | "
                          f"Align Cos: {loss_latent_align.item():.4f} | "
                          f"L1 Token Match: {token_match_l1:4.1f}% | "
                          f"Tokens L1: {unique_c}/{num_embeddings} ({pct_u:.1f}%) | Perp: {perp:.1f} | {elapsed:.1f}s", flush=True)[cite: 50]
                running_loss_g = 0.0[cite: 50]
                step_start_time = time.time()[cite: 50]

        scheduler_g.step()[cite: 50]
        if use_discriminator:[cite: 50]
            scheduler_d.step()[cite: 50]

        if rank == 0:[cite: 50]
            avg_epoch_g = epoch_loss_g / total_steps[cite: 50]
            total_epoch_time = time.time() - epoch_start_time[cite: 50]
            print(f"📢 Epoch {epoch:03d} terminata in {total_epoch_time:.1f}s ({total_epoch_time/60:.2f} min). Media G-Loss: {avg_epoch_g:.5f}\n", flush=True)[cite: 50]

            if (epoch + 1) % 5 == 0 or epoch == (end_epoch - 1):[cite: 50]
                checkpoint_path = os.path.join(target_model_dir, f"dual_vae_epoch_{epoch}.pt")[cite: 50]
                torch.save({[cite: 50]
                    'epoch': epoch,[cite: 50]
                    'mel_vae_state_dict': mel_vae.module.state_dict(),[cite: 50]
                    'octave_vae_state_dict': octave_vae.module.state_dict(),[cite: 50]
                    'discriminator_state_dict': discriminator.module.state_dict(),[cite: 50]
                    'optimizer_state_dict': optimizer_g.state_dict(),[cite: 50]
                    'scheduler_state_dict': scheduler_g.state_dict(),[cite: 50]
                    'loss': avg_epoch_g,[cite: 50]
                    'bottleneck_type': DEFAULT_BOTTLENECK_TYPE,
                    'latent_dim': latent_dim,[cite: 50]
                    'num_embeddings': num_embeddings,[cite: 50]
                    'num_quantizers': num_quantizers,
                }, checkpoint_path)[cite: 50]
                print(f"💾 Checkpoint Dual RVQ-GAN salvato in: {checkpoint_path}", flush=True)

    dataset.close()[cite: 50]
    cleanup_distributed_environment(rank)[cite: 50]

if __name__ == "__main__":[cite: 50]
    main()[cite: 50]

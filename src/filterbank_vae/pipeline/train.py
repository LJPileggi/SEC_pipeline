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
from filterbank_vae.pipeline.spectral import OnlineSpectrogramPipeline, AcousticStructuralLoss, feature_matching_loss
from filterbank_vae.models.vae import MelVQVAE, OctaveVQVAE, DiscriminatorWithFeatures, ClassAuxiliaryHead, DEFAULT_BOTTLENECK_TYPE

DEFAULT_EPOCHS = 50
LOCAL_BATCH_SIZE = 8
ACCUMULATION_STEPS = 2
LATENT_DIM = 256
NUM_EMBEDDINGS = 1024
NUM_QUANTIZERS = 4
DISC_START_EPOCH = int(os.environ.get("DISC_START_EPOCH", 5))
DISC_WEIGHT = float(os.environ.get("DISC_WEIGHT", 0.1))
CLS_WEIGHT = float(os.environ.get("CLS_WEIGHT", 1.0))


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Dual RVQ-GAN Training: Unconditioned Decoder + Class-Supervised Bottleneck & Discriminator")
    parser.add_argument("--start_epoch", type=int, default=int(os.environ.get("START_EPOCH", 0)))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("NUM_EPOCHS", DEFAULT_EPOCHS)))
    parser.add_argument("--bottleneck_type", type=str, default=DEFAULT_BOTTLENECK_TYPE)
    parser.add_argument("--latent_dim", type=int, default=LATENT_DIM)
    parser.add_argument("--num_embeddings", type=int, default=NUM_EMBEDDINGS)
    parser.add_argument("--num_quantizers", type=int, default=NUM_QUANTIZERS)
    parser.add_argument("--cls_weight", type=float, default=CLS_WEIGHT)
    return parser.parse_args()


def compute_cosine_alignment_loss(z_e_oct, z_e_mel):
    cos_sim = F.cosine_similarity(z_e_oct, z_e_mel, dim=1)
    return torch.mean(1.0 - cos_sim)


def compute_codebook_diagnostics(indices, num_embeddings):
    first_level = indices[:, 0, :, :].flatten()
    unique_count = torch.unique(first_level).numel()
    pct_usage = (unique_count / num_embeddings) * 100.0

    encodings = F.one_hot(first_level, num_classes=num_embeddings).float()
    avg_probs = encodings.mean(dim=0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))).item()
    return unique_count, pct_usage, perplexity


def main():
    args = parse_args()
    start_epoch = args.start_epoch
    num_epochs = args.epochs
    end_epoch = start_epoch + num_epochs
    latent_dim = args.latent_dim
    num_embeddings = args.num_embeddings
    num_quantizers = args.num_quantizers

    rank, world_size = setup_environ_vars(slurm=True)
    device = setup_distributed_environment(rank, world_size, slurm=True)

    classes_list, _, _, _, sampling_rate, _, _, seed, _, _, _ = get_config_from_yaml("config0.yaml")
    num_classes = len(classes_list)

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
    dataloader = DataLoader(dataset, batch_size=LOCAL_BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)

    # VAE puro incondizionato
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

    # Testa ausiliaria sui token (solo training)
    aux_cls_head = ClassAuxiliaryHead(latent_dim=latent_dim, num_classes=num_classes).to(device)

    # Projection Discriminator condizionale (solo training)
    discriminator = DiscriminatorWithFeatures(in_channels=1, ndf=64, num_classes=num_classes).to(device)

    gen_params = list(mel_vae.parameters()) + list(octave_vae.parameters()) + list(aux_cls_head.parameters())
    optimizer_g = torch.optim.AdamW(gen_params, lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-5)
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=end_epoch, eta_min=1e-6)

    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-5)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=end_epoch, eta_min=1e-6)

    base_model_dir = os.environ.get("MODEL_CHECKPOINT_DIR", os.path.join(src_root, ".models", "vae_model"))
    target_model_dir = base_model_dir if base_model_dir.startswith("/") else os.path.join(src_root, base_model_dir)
    os.makedirs(target_model_dir, exist_ok=True)

    if start_epoch > 0:
        prev_epoch = start_epoch - 1
        ckpt_candidate = os.path.join(target_model_dir, f"dual_vae_epoch_{prev_epoch}.pt")
        if not os.path.exists(ckpt_candidate) and os.environ.get("MODELS_GLOBAL"):
            ckpt_candidate = os.path.join(os.environ.get("MODELS_GLOBAL"), f"dual_vae_epoch_{prev_epoch}.pt")

        if os.path.exists(ckpt_candidate):
            if rank == 0:
                print(f"🔄 [RESUMPTION] Caricamento checkpoint Dual RVQ-GAN: {ckpt_candidate}", flush=True)
            checkpoint = torch.load(ckpt_candidate, map_location=device)
            mel_vae.load_state_dict(checkpoint['mel_vae_state_dict'], strict=False)
            octave_vae.load_state_dict(checkpoint['octave_vae_state_dict'], strict=False)
            if 'discriminator_state_dict' in checkpoint:
                discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=False)
            if 'aux_cls_head_state_dict' in checkpoint:
                aux_cls_head.load_state_dict(checkpoint['aux_cls_head_state_dict'])
            optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler_g.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            if rank == 0:
                print(f"❌ [CRITICAL] Checkpoint non trovato: {ckpt_candidate}", flush=True)
            sys.exit(1)

    mel_vae = nn.parallel.DistributedDataParallel(mel_vae, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    octave_vae = nn.parallel.DistributedDataParallel(octave_vae, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    aux_cls_head = nn.parallel.DistributedDataParallel(aux_cls_head, device_ids=[rank], output_device=rank)
    discriminator = nn.parallel.DistributedDataParallel(discriminator, device_ids=[rank], output_device=rank, find_unused_parameters=True, broadcast_buffers=False)

    acoustic_loss_fn = AcousticStructuralLoss(alpha=1.0, beta=0.5).to(device)

    total_steps = len(dataloader)
    print_freq = max(1, total_steps // 10)

    if rank == 0:
        print("==========================================================", flush=True)
        print(f"🏁 Dual RVQ-GAN Training Initialized (DDP, GPUs: {world_size})", flush=True)
        print(f"   • MelDecoder: Incondizionato (Pure Latent Decoding)", flush=True)
        print(f"   • Discriminatore: Projection Discriminator Condizionale su Classe", flush=True)
        print(f"   • Supervisione Semantica Token: GAP+GMP Auxiliary Head (Weight: {args.cls_weight})", flush=True)
        print(f"   • Alignment: Continuous Point-wise Cosine Distance (Weight: 3.0)", flush=True)
        print(f"   • MelDecoder Protection: Frozen during cross-reconstruction", flush=True)
        print(f"   • Adversarial Warm-up: Da Epoca {DISC_START_EPOCH}", flush=True)
        print("==========================================================\n", flush=True)

    optimizer_g.zero_grad(set_to_none=True)
    optimizer_d.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, end_epoch):
        mel_vae.train()
        octave_vae.train()
        aux_cls_head.train()
        discriminator.train()
        sampler.set_epoch(epoch)

        epoch_loss_g = 0.0
        running_loss_g = 0.0
        epoch_start_time = time.time()
        step_start_time = time.time()

        use_discriminator = (epoch >= DISC_START_EPOCH)

        for step, (raw_audio, class_labels) in enumerate(dataloader):
            raw_audio = raw_audio.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)

            if epoch < 10:
                fraction_id = np.random.choice([6, 12, 16])
            elif epoch < 25:
                fraction_id = np.random.choice([3, 6, 12, 16, 24])
            else:
                fraction_id = np.random.choice([1, 3, 6, 12, 16, 24, 32])

            x_mel_target, x_octave_pchip = spectral_pipeline(raw_audio, format_id=1, fraction_id=fraction_id, device=device)

            # ------------------------------------------------------------------
            # 1. PASSO GENERATORE
            # ------------------------------------------------------------------
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # 1a. Mel Branch (Decoder incondizionato)
                recon_mel, vq_loss_mel, z_e_mel, z_q_mel, indices_mel = mel_vae(x_mel_target)

                # 1b. Octave Branch
                recon_oct, vq_loss_oct, z_e_oct, z_q_oct, indices_oct = octave_vae(x_octave_pchip)

                loss_rec_mel = acoustic_loss_fn(recon_mel, x_mel_target)
                loss_rec_oct = F.l1_loss(recon_oct, x_octave_pchip)
                loss_vq = vq_loss_mel + vq_loss_oct

                # 1c. Supervisione Semantica dei Token (GAP + GMP Head)
                logits_mel = aux_cls_head(z_q_mel)
                logits_oct = aux_cls_head(z_q_oct)
                loss_cls = 0.5 * (F.cross_entropy(logits_mel, class_labels) + F.cross_entropy(logits_oct, class_labels))

                # 1d. Allineamento continuo degli spazi latenti
                loss_align_cos = compute_cosine_alignment_loss(z_e_oct, z_e_mel.detach())

                # 1e. Cross-Domain Reconstruction incondizionata (MelDecoder protetto)
                for param in mel_vae.module.decoder.parameters():
                    param.requires_grad = False
                cross_mel_recon = mel_vae.module.decoder(z_q_oct)
                for param in mel_vae.module.decoder.parameters():
                    param.requires_grad = True

                loss_cross = acoustic_loss_fn(cross_mel_recon, x_mel_target)

                loss_g_base = (loss_rec_mel + loss_rec_oct) + \
                              loss_vq + \
                              (args.cls_weight * loss_cls) + \
                              3.0 * loss_align_cos + \
                              1.0 * loss_cross

                loss_adv_total = torch.tensor(0.0, device=device)
                loss_fm_total = torch.tensor(0.0, device=device)

                if use_discriminator:
                    # Il discriminatore viene interrogato fornendogli la classe reale del batch
                    pred_fake_mel, feats_fake_mel = discriminator(recon_mel, class_idx=class_labels)
                    pred_fake_cross, feats_fake_cross = discriminator(cross_mel_recon, class_idx=class_labels)

                    with torch.no_grad():
                        _, feats_real_mel = discriminator(x_mel_target, class_idx=class_labels)

                    loss_fm_mel = feature_matching_loss(feats_real_mel, feats_fake_mel)
                    loss_fm_cross = feature_matching_loss(feats_real_mel, feats_fake_cross)
                    loss_fm_total = 0.5 * (loss_fm_mel + loss_fm_cross)

                    loss_adv_mel = -torch.mean(pred_fake_mel)
                    loss_adv_cross = -torch.mean(pred_fake_cross)
                    loss_adv_total = 0.5 * (loss_adv_mel + loss_adv_cross)

                    loss_g_total = loss_g_base + (0.5 * loss_fm_total) + (DISC_WEIGHT * loss_adv_total)
                else:
                    loss_g_total = loss_g_base

                loss_g_scaled = loss_g_total / ACCUMULATION_STEPS

            loss_g_scaled.backward()

            # ------------------------------------------------------------------
            # 2. PASSO DISCRIMINATORE (CONDIZIONALE SU CLASSE)
            # ------------------------------------------------------------------
            loss_d_total = torch.tensor(0.0, device=device)
            if use_discriminator:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    pred_real, _ = discriminator(x_mel_target, class_idx=class_labels)
                    pred_fake_mel, _ = discriminator(recon_mel.detach(), class_idx=class_labels)
                    pred_fake_cross, _ = discriminator(cross_mel_recon.detach(), class_idx=class_labels)

                    loss_d_real = torch.mean(F.relu(1.0 - pred_real))
                    loss_d_fake = 0.5 * (torch.mean(F.relu(1.0 + pred_fake_mel)) + torch.mean(F.relu(1.0 + pred_fake_cross)))
                    loss_d_total = 0.5 * (loss_d_real + loss_d_fake)
                    loss_d_scaled = loss_d_total / ACCUMULATION_STEPS

                loss_d_scaled.backward()

            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == total_steps:
                torch.nn.utils.clip_grad_norm_(gen_params, max_norm=1.0)
                optimizer_g.step()
                optimizer_g.zero_grad(set_to_none=True)

                if use_discriminator:
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    optimizer_d.step()
                    optimizer_d.zero_grad(set_to_none=True)

            current_loss_g = loss_g_total.item()
            if math.isnan(current_loss_g) or math.isinf(current_loss_g):
                if rank == 0:
                    print(f"\n❌ [CRITICAL] Loss NaN/Inf a Epoca {epoch}, Step {step}/{total_steps}!")
                sys.exit(1)

            epoch_loss_g += current_loss_g
            running_loss_g += current_loss_g

            if (step + 1) % print_freq == 0 or (step + 1) == total_steps:
                unique_c, pct_u, perp = compute_codebook_diagnostics(indices_mel, num_embeddings)
                token_match_l1 = (indices_mel[:, 0] == indices_oct[:, 0]).float().mean().item() * 100.0
                acc_mel = (logits_mel.argmax(dim=-1) == class_labels).float().mean().item() * 100.0
                acc_oct = (logits_oct.argmax(dim=-1) == class_labels).float().mean().item() * 100.0

                if rank == 0:
                    elapsed = time.time() - step_start_time
                    percent = ((step + 1) / total_steps) * 100
                    avg_g = running_loss_g / print_freq
                    print(f" ⏱️  Epoch [{epoch:03d}/{end_epoch-1:03d}] | Progress: {percent:5.1f}% | "
                          f"G-Loss: {avg_g:.4f} | D-Loss: {loss_d_total.item():.4f} | "
                          f"ClsLoss: {loss_cls.item():.3f} (Acc Mel: {acc_mel:.1f}%, Oct: {acc_oct:.1f}%) | "
                          f"L1 Token Match: {token_match_l1:4.1f}% | {elapsed:.1f}s", flush=True)
                running_loss_g = 0.0
                step_start_time = time.time()

        scheduler_g.step()
        if use_discriminator:
            scheduler_d.step()

        if rank == 0:
            avg_epoch_g = epoch_loss_g / total_steps
            total_epoch_time = time.time() - epoch_start_time
            print(f"📢 Epoch {epoch:03d} terminata in {total_epoch_time:.1f}s ({total_epoch_time/60:.2f} min). Media G-Loss: {avg_epoch_g:.5f}\n", flush=True)

            if (epoch + 1) % 5 == 0 or epoch == (end_epoch - 1):
                checkpoint_path = os.path.join(target_model_dir, f"dual_vae_epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'mel_vae_state_dict': mel_vae.module.state_dict(),
                    'octave_vae_state_dict': octave_vae.module.state_dict(),
                    'aux_cls_head_state_dict': aux_cls_head.module.state_dict(),
                    'discriminator_state_dict': discriminator.module.state_dict(),
                    'optimizer_state_dict': optimizer_g.state_dict(),
                    'scheduler_state_dict': scheduler_g.state_dict(),
                    'loss': avg_epoch_g,
                    'bottleneck_type': DEFAULT_BOTTLENECK_TYPE,
                    'latent_dim': latent_dim,
                    'num_embeddings': num_embeddings,
                    'num_quantizers': num_quantizers,
                    'num_classes': num_classes,
                }, checkpoint_path)
                print(f"💾 Checkpoint Dual RVQ-GAN salvato in: {checkpoint_path}", flush=True)

    dataset.close()
    cleanup_distributed_environment(rank)


if __name__ == "__main__":
    main()

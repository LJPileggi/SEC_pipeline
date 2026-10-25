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

# ESATTAMENTE DUE LIVELLI: da pipeline/ a filterbank_vae/, da filterbank_vae/ alla radice src/
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
DISC_WEIGHT = float(os.environ.get("DISC_WEIGHT", 0.05))

# PESI FISSI COSTANTI: Nessun curriculum dinamico che sbilancia l'equilibrio
CLS_WEIGHT = float(os.environ.get("CLS_WEIGHT", 1.0))
CODEBOOK_DISTILL_WEIGHT = float(os.environ.get("CODEBOOK_DISTILL_WEIGHT", 1.0))
ENTROPY_WEIGHT = float(os.environ.get("ENTROPY_WEIGHT", 0.01))


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Dual RVQ-GAN Reproducible Training: Dual-Phase Annealing & Stable Floor")
    parser.add_argument("--start_epoch", type=int, default=int(os.environ.get("START_EPOCH", 0)))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("NUM_EPOCHS", DEFAULT_EPOCHS)))
    parser.add_argument("--total_planned_epochs", type=int, default=int(os.environ.get("TOTAL_PLANNED_EPOCHS", 50)))
    parser.add_argument("--bottleneck_type", type=str, default=DEFAULT_BOTTLENECK_TYPE)
    parser.add_argument("--latent_dim", type=int, default=LATENT_DIM)
    parser.add_argument("--num_embeddings", type=int, default=NUM_EMBEDDINGS)
    parser.add_argument("--num_quantizers", type=int, default=NUM_QUANTIZERS)
    parser.add_argument("--cls_weight", type=float, default=CLS_WEIGHT)
    parser.add_argument("--codebook_distill_weight", type=float, default=CODEBOOK_DISTILL_WEIGHT)
    parser.add_argument("--entropy_weight", type=float, default=ENTROPY_WEIGHT)
    parser.add_argument("--warm_restart", action="store_true", default=True, help="Supporto compatibilità flag run script")
    parser.add_argument("--freeze_lr", action="store_true", default=False, help="Flag compatibilità retroattiva")
    return parser.parse_args()


def get_reproducible_lr_factor(current_epoch, total_epochs=50, split_epoch=35):
    """
    Funzione deterministica per la curva LR:
    - Fase 1 (0 -> 34): Cosine Annealing da 1.0 (2.0e-4) fino a ~0.475 (9.5e-5).
    - Fase 2 (35 -> 49): Warm-Bump al 65% (1.3e-4) con discesa morbida fino al floor di 0.175 (3.5e-5).
    Garantisce la medesima traiettoria sia partendo da 0 sia riprendendo da 35.
    """
    if current_epoch < split_epoch:
        progress = float(current_epoch) / float(max(1, split_epoch))
        factor = 0.475 + 0.525 * 0.5 * (1.0 + math.cos(progress * math.pi))
    else:
        rem_epochs = max(1, total_epochs - split_epoch)
        progress = float(current_epoch - split_epoch) / float(rem_epochs)
        factor = 0.175 + (0.650 - 0.175) * 0.5 * (1.0 + math.cos(progress * math.pi))
    return factor


def compute_cosine_alignment_loss(z_e_oct, z_e_mel):
    cos_sim = F.cosine_similarity(z_e_oct, z_e_mel, dim=1)
    return torch.mean(1.0 - cos_sim)


def compute_codebook_distillation_loss(quantizer, z_e_oct, z_e_mel, temperature=0.1):
    first_stage = quantizer.layers[0]
    sim_mel, _ = first_stage.get_codebook_similarities(z_e_mel.detach())
    sim_oct, _ = first_stage.get_codebook_similarities(z_e_oct)

    prob_mel = F.softmax(sim_mel / temperature, dim=-1)
    log_prob_oct = F.log_softmax(sim_oct / temperature, dim=-1)

    kl_div = F.kl_div(log_prob_oct, prob_mel, reduction="batchmean") * (temperature ** 2)
    return kl_div


def compute_batch_entropy_loss(quantizer, z_e_oct, num_embeddings=1024, temperature=0.2):
    first_stage = quantizer.layers[0]
    sim_oct, _ = first_stage.get_codebook_similarities(z_e_oct)
    prob_oct = F.softmax(sim_oct / temperature, dim=-1)

    avg_probs = torch.mean(prob_oct, dim=0)
    entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-6))
    max_entropy = math.log(num_embeddings)
    loss_entropy = (max_entropy - entropy) / max_entropy
    return loss_entropy


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
    total_planned = args.total_planned_epochs
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

    aux_cls_head = ClassAuxiliaryHead(latent_dim=latent_dim, num_classes=num_classes).to(device)
    discriminator = DiscriminatorWithFeatures(in_channels=1, ndf=64, num_classes=num_classes).to(device)

    gen_params = list(mel_vae.parameters()) + list(octave_vae.parameters()) + list(aux_cls_head.parameters())

    # -------------------------------------------------------------------------
    # SCHEDULER UNIFICATO E DETERMINISTICO BASATO SU EPOCA ASSOLUTA
    # -------------------------------------------------------------------------
    initial_lr_g = 2.0e-4
    initial_lr_d = 1.0e-4

    optimizer_g = torch.optim.AdamW(gen_params, lr=initial_lr_g, betas=(0.5, 0.9), weight_decay=1e-5)
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(
        optimizer_g,
        lr_lambda=lambda ep: get_reproducible_lr_factor(ep, total_epochs=total_planned, split_epoch=35)
    )

    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=initial_lr_d, betas=(0.5, 0.9), weight_decay=1e-5)
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(
        optimizer_d,
        lr_lambda=lambda ep: get_reproducible_lr_factor(ep, total_epochs=total_planned, split_epoch=35)
    )

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
                print(f"🔄 [RESUMPTION] Caricamento pesi modello da: {ckpt_candidate}", flush=True)
            checkpoint = torch.load(ckpt_candidate, map_location=device)
            mel_vae.load_state_dict(checkpoint['mel_vae_state_dict'], strict=False)
            octave_vae.load_state_dict(checkpoint['octave_vae_state_dict'], strict=False)
            if 'discriminator_state_dict' in checkpoint:
                discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=False)
            if 'aux_cls_head_state_dict' in checkpoint:
                try:
                    aux_cls_head.load_state_dict(checkpoint['aux_cls_head_state_dict'], strict=True)
                except Exception:
                    if rank == 0:
                        print("⚠️ [NOTE] Nuova TemporalClassAuxiliaryHead inizializzata da zero.", flush=True)

            # Allinea lo scheduler globale all'indice esatto dell'epoca da cui si riparte
            for _ in range(start_epoch):
                scheduler_g.step()
                scheduler_d.step()

            current_lr_g = scheduler_g.get_last_lr()[0]
            if rank == 0:
                print(f"🚀 [ALIGNED-RESTART] Sincronizzazione scheduler a Epoca {start_epoch} (G-LR: {current_lr_g:.2e})", flush=True)
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
        print(f"🏁 Dual RVQ-GAN Unified Reproducible Training (DDP, GPUs: {world_size})", flush=True)
        print(f"   • Range Epoche: [{start_epoch} -> {end_epoch - 1}] (Pianificate totali: {total_planned})", flush=True)
        print(f"   • Curva LR: Dual-Phase Deterministica (Fase 1: 0-34, Fase 2: 35-49 floor 3.5e-5)", flush=True)
        print(f"   • Pesi Loss Costanti: Cls={args.cls_weight}, Distill={args.codebook_distill_weight}, CosAlign=3.0", flush=True)
        print(f"   • Codebook EMA Schedule: decay=0.99 (ep < 25), decay=0.999 (ep >= 25)", flush=True)
        print("==========================================================\n", flush=True)

    optimizer_g.zero_grad(set_to_none=True)
    optimizer_d.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, end_epoch):
        # Stabilizzazione deterministica del Codebook EMA
        if epoch >= 25:
            mel_vae.module.quantizer.set_decay(0.999)
        else:
            mel_vae.module.quantizer.set_decay(0.99)

        mel_vae.train()
        octave_vae.train()
        aux_cls_head.train()
        discriminator.train()
        sampler.set_epoch(epoch)

        epoch_loss_g = 0.0
        running_loss_g = 0.0

        running_rec_mel = 0.0
        running_rec_oct = 0.0
        running_cross = 0.0
        running_align = 0.0
        running_vq = 0.0
        running_adv = 0.0

        epoch_start_time = time.time()
        step_start_time = time.time()

        use_discriminator = (epoch >= DISC_START_EPOCH)
        if use_discriminator:
            ramp_factor = min(1.0, float(epoch - DISC_START_EPOCH + 1) / 3.0)
            cur_disc_weight = DISC_WEIGHT * ramp_factor
        else:
            cur_disc_weight = 0.0

        for step, (raw_audio, class_labels) in enumerate(dataloader):
            raw_audio = raw_audio.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)

            if epoch < 10:
                fraction_id = int(np.random.choice([6, 12, 16]))
            elif epoch < 25:
                fraction_id = int(np.random.choice([3, 6, 12, 16, 24]))
            else:
                fraction_id = int(np.random.choice([1, 3, 6, 12, 16, 24, 32]))

            x_mel_target, x_octave_pchip = spectral_pipeline(raw_audio, format_id=1, fraction_id=fraction_id, device=device)

            oct_rec_weight = 0.25 if fraction_id == 1 else (0.5 if fraction_id == 3 else 1.0)

            # ------------------------------------------------------------------
            # 1. PASSO GENERATORE
            # ------------------------------------------------------------------
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                recon_mel, vq_loss_mel, z_e_mel, z_q_mel, indices_mel = mel_vae(x_mel_target)
                recon_oct, vq_loss_oct, z_e_oct, z_q_oct, indices_oct = octave_vae(x_octave_pchip)

                loss_rec_mel = acoustic_loss_fn(recon_mel, x_mel_target)
                loss_rec_oct = F.l1_loss(recon_oct, x_octave_pchip) * oct_rec_weight
                loss_vq = vq_loss_mel + vq_loss_oct

                logits_mel_t = aux_cls_head(z_q_mel)
                logits_oct_t = aux_cls_head(z_q_oct)

                targets_t = class_labels.unsqueeze(1).expand(-1, 72)
                loss_cls_mel = F.cross_entropy(logits_mel_t, targets_t)
                loss_cls_oct = F.cross_entropy(logits_oct_t, targets_t)
                loss_cls = 0.5 * (loss_cls_mel + loss_cls_oct)

                loss_align_cos = compute_cosine_alignment_loss(z_e_oct, z_e_mel.detach())

                loss_codebook_distill = compute_codebook_distillation_loss(
                    mel_vae.module.quantizer, z_e_oct, z_e_mel, temperature=0.1
                )

                loss_entropy = compute_batch_entropy_loss(
                    mel_vae.module.quantizer, z_e_oct, num_embeddings=num_embeddings, temperature=0.2
                )

                cross_mel_recon = mel_vae.module.decoder(z_q_oct)
                loss_cross = acoustic_loss_fn(cross_mel_recon, x_mel_target)

                loss_g_base = (loss_rec_mel + loss_rec_oct) + \
                              loss_vq + \
                              (args.cls_weight * loss_cls) + \
                              (3.0 * loss_align_cos) + \
                              (args.codebook_distill_weight * loss_codebook_distill) + \
                              (args.entropy_weight * loss_entropy) + \
                              (1.0 * loss_cross)

                loss_adv_total = torch.tensor(0.0, device=device)
                loss_fm_total = torch.tensor(0.0, device=device)

                if use_discriminator:
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

                    loss_g_total = loss_g_base + (0.5 * loss_fm_total) + (cur_disc_weight * loss_adv_total)
                else:
                    loss_g_total = loss_g_base

                loss_g_scaled = loss_g_total / ACCUMULATION_STEPS

            loss_g_scaled.backward()

            # ------------------------------------------------------------------
            # 2. PASSO DISCRIMINATORE
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

            norm_g_oct = 0.0
            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == total_steps:
                grads_oct = [p.grad.detach() for p in octave_vae.parameters() if p.grad is not None]
                if grads_oct:
                    norm_g_oct = torch.norm(torch.stack([torch.norm(g, 2) for g in grads_oct]), 2).item()

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

            running_rec_mel += loss_rec_mel.item()
            running_rec_oct += loss_rec_oct.item()
            running_cross += loss_cross.item()
            running_align += (3.0 * loss_align_cos).item()
            running_vq += loss_vq.item()
            running_adv += (cur_disc_weight * loss_adv_total).item() if use_discriminator else 0.0

            if (step + 1) % print_freq == 0 or (step + 1) == total_steps:
                unique_c, pct_u, perp = compute_codebook_diagnostics(indices_mel, num_embeddings)
                token_match_l1 = (indices_mel[:, 0] == indices_oct[:, 0]).float().mean().item() * 100.0

                current_lr = scheduler_g.get_last_lr()[0]
                if rank == 0:
                    elapsed = time.time() - step_start_time
                    percent = ((step + 1) / total_steps) * 100
                    avg_g = running_loss_g / print_freq
                    d_val = loss_d_total.item() if use_discriminator else 0.0

                    m_rec_mel = running_rec_mel / print_freq
                    m_rec_oct = running_rec_oct / print_freq
                    m_cross = running_cross / print_freq
                    m_align = running_align / print_freq
                    m_vq = running_vq / print_freq
                    m_adv = running_adv / print_freq

                    print(f" ⏱️ Ep [{epoch:02d}] {percent:4.1f}% | G: {avg_g:.3f} | "
                          f"[RecM: {m_rec_mel:.3f} | RecOct: {m_rec_oct:.3f} | Cross: {m_cross:.3f} | Cos: {m_align:.3f} | VQ: {m_vq:.3f} | Adv: {m_adv:.3f}] | "
                          f"Cls: {loss_cls.item():.3f} | L1: {token_match_l1:4.1f}% | ||g_oct||: {norm_g_oct:.2f} | LR: {current_lr:.1e} | {elapsed:.1f}s", flush=True)

                running_loss_g = 0.0
                running_rec_mel = 0.0
                running_rec_oct = 0.0
                running_cross = 0.0
                running_align = 0.0
                running_vq = 0.0
                running_adv = 0.0
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

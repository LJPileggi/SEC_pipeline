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
from filterbank_vae.pipeline.spectral import OnlineSpectrogramPipeline, feature_matching_loss
from filterbank_vae.models.vae import MelVQVAE, OctaveVQVAE, DiscriminatorWithFeatures, ClassAuxiliaryHead, DEFAULT_BOTTLENECK_TYPE

DEFAULT_EPOCHS = 75
TOTAL_PLANNED_EPOCHS = 75
SPLIT_EPOCH = 50
LOCAL_BATCH_SIZE = 8
ACCUMULATION_STEPS = 2
LATENT_DIM = 256
NUM_EMBEDDINGS = 1024
NUM_QUANTIZERS = 4

# IPERPARAMETRI CALIBRATI V2 (Anti-Blur & Anti-Overlap)
COMMITMENT_COST = float(os.environ.get("COMMITMENT_COST", 0.25))
DISC_START_EPOCH = int(os.environ.get("DISC_START_EPOCH", 5))
DISC_WEIGHT = float(os.environ.get("DISC_WEIGHT", 0.50)) # Rialzato di 10x per forzare armoniche nitide
FM_WEIGHT = float(os.environ.get("FM_WEIGHT", 1.0))

CLS_WEIGHT = float(os.environ.get("CLS_WEIGHT", 1.0))
COS_ALIGN_WEIGHT = float(os.environ.get("COS_ALIGN_WEIGHT", 6.0))
CODEBOOK_DISTILL_WEIGHT = float(os.environ.get("CODEBOOK_DISTILL_WEIGHT", 1.5))
ENTROPY_WEIGHT = float(os.environ.get("ENTROPY_WEIGHT", 0.02))


class MultiScaleAcousticLoss(nn.Module):
    """
    Funzione di costo acustica anti-sfocatura:
    1) L1 su spettrogramma;
    2) Gradient/Edge Loss orizzontale (tempo) e verticale (frequenza) per preservare fronti d'onda e pettini;
    3) Log-Spectral Difference per penalizzare l'appiattimento nelle bande deboli.
    """
    def __init__(self, l1_weight=1.0, grad_weight=1.5, log_mag_weight=0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.grad_weight = grad_weight
        self.log_mag_weight = log_mag_weight

    def forward(self, pred, target):
        loss_l1 = F.l1_loss(pred, target)

        diff_f_target = target[:, :, 1:, :] - target[:, :, :-1, :]
        diff_f_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        loss_grad_f = F.l1_loss(diff_f_pred, diff_f_target)

        diff_t_target = target[:, :, :, 1:] - target[:, :, :, :-1]
        diff_t_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        loss_grad_t = F.l1_loss(diff_t_pred, diff_t_target)

        eps = 1e-4
        log_pred = torch.log(torch.clamp(pred, min=0.0) + eps)
        log_target = torch.log(torch.clamp(target, min=0.0) + eps)
        loss_log = F.l1_loss(log_pred, log_target)

        return self.l1_weight * loss_l1 + self.grad_weight * (loss_grad_f + loss_grad_t) + self.log_mag_weight * loss_log


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Dual RVQ-GAN Calibrated V2 (Grid 16x144, High-Disc, MultiScale Loss)")
    parser.add_argument("--start_epoch", type=int, default=int(os.environ.get("START_EPOCH", 0)))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("NUM_EPOCHS", DEFAULT_EPOCHS)))
    parser.add_argument("--total_planned_epochs", type=int, default=int(os.environ.get("TOTAL_PLANNED_EPOCHS", TOTAL_PLANNED_EPOCHS)))
    parser.add_argument("--split_epoch", type=int, default=int(os.environ.get("SPLIT_EPOCH", SPLIT_EPOCH)))
    parser.add_argument("--bottleneck_type", type=str, default=DEFAULT_BOTTLENECK_TYPE)
    parser.add_argument("--latent_dim", type=int, default=LATENT_DIM)
    parser.add_argument("--num_embeddings", type=int, default=NUM_EMBEDDINGS)
    parser.add_argument("--num_quantizers", type=int, default=NUM_QUANTIZERS)
    parser.add_argument("--commitment_cost", type=float, default=COMMITMENT_COST)
    parser.add_argument("--cos_align_weight", type=float, default=COS_ALIGN_WEIGHT)
    parser.add_argument("--cls_weight", type=float, default=CLS_WEIGHT)
    parser.add_argument("--codebook_distill_weight", type=float, default=CODEBOOK_DISTILL_WEIGHT)
    parser.add_argument("--entropy_weight", type=float, default=ENTROPY_WEIGHT)
    parser.add_argument("--disc_weight", type=float, default=DISC_WEIGHT)
    parser.add_argument("--warm_restart", action="store_true", default=True)
    return parser.parse_args()


def get_reproducible_lr_factor(current_epoch, total_epochs=75, split_epoch=50):
    if current_epoch < split_epoch:
        progress = float(current_epoch) / float(max(1, split_epoch))
        factor = 0.475 + 0.525 * 0.5 * (1.0 + math.cos(progress * math.pi))
    else:
        rem_epochs = max(1, total_epochs - split_epoch)
        progress = float(current_epoch - split_epoch) / float(rem_epochs)
        factor = 0.175 + (0.650 - 0.175) * 0.5 * (1.0 + math.cos(progress * math.pi))
    return factor


def compute_frequency_weighted_cosine_loss(z_e_oct, z_e_mel):
    """
    Allineamento Coseno pesato in frequenza su griglia [B, D, 16, 144].
    Bande 0..7 (basse/medie): peso 1.0.
    Bande 8..15 (alte/transienti): peso 2.5 per contrastare il disallineamento evidenziato nei test.
    """
    cos_sim = F.cosine_similarity(z_e_oct, z_e_mel, dim=1) # [B, 16, 144]
    diff = 1.0 - cos_sim

    H = diff.shape[1]
    weights = torch.ones(H, device=diff.device)
    split_h = H // 2
    weights[split_h:] = 2.5 # enfasi alte frequenze
    weights = weights.view(1, H, 1)

    weighted_diff = diff * weights
    return torch.mean(weighted_diff)


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
    split_epoch = args.split_epoch
    latent_dim = args.latent_dim
    num_embeddings = args.num_embeddings
    num_quantizers = args.num_quantizers
    commitment_cost = args.commitment_cost
    cos_align_weight = args.cos_align_weight
    disc_weight = args.disc_weight

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
        num_quantizers=num_quantizers,
        commitment_cost=commitment_cost
    ).to(device)

    octave_vae = OctaveVQVAE(
        base_channels=64,
        latent_dim=latent_dim,
        num_embeddings=num_embeddings,
        num_quantizers=num_quantizers,
        commitment_cost=commitment_cost,
        shared_quantizer=mel_vae.quantizer
    ).to(device)

    aux_cls_head = ClassAuxiliaryHead(latent_dim=latent_dim, num_classes=num_classes).to(device)
    discriminator = DiscriminatorWithFeatures(in_channels=1, ndf=64, num_classes=num_classes).to(device)

    gen_params = list(mel_vae.parameters()) + list(octave_vae.parameters()) + list(aux_cls_head.parameters())

    initial_lr_g = 2.0e-4
    initial_lr_d = 1.0e-4

    optimizer_g = torch.optim.AdamW(gen_params, lr=initial_lr_g, betas=(0.5, 0.9), weight_decay=1e-5)
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(
        optimizer_g,
        lr_lambda=lambda ep: get_reproducible_lr_factor(ep, total_epochs=total_planned, split_epoch=split_epoch)
    )

    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=initial_lr_d, betas=(0.5, 0.9), weight_decay=1e-5)
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(
        optimizer_d,
        lr_lambda=lambda ep: get_reproducible_lr_factor(ep, total_epochs=total_planned, split_epoch=split_epoch)
    )

    base_model_dir = os.environ.get("MODEL_CHECKPOINT_DIR", os.path.join(src_root, ".models", "vae_model"))
    target_model_dir = base_model_dir if base_model_dir.startswith("/") else os.path.join(src_root, base_model_dir)
    os.makedirs(target_model_dir, exist_ok=True)

    mel_vae = nn.parallel.DistributedDataParallel(mel_vae, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    octave_vae = nn.parallel.DistributedDataParallel(octave_vae, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    aux_cls_head = nn.parallel.DistributedDataParallel(aux_cls_head, device_ids=[rank], output_device=rank)
    discriminator = nn.parallel.DistributedDataParallel(discriminator, device_ids=[rank], output_device=rank, find_unused_parameters=True, broadcast_buffers=False)

    acoustic_loss_fn = MultiScaleAcousticLoss(l1_weight=1.0, grad_weight=1.5, log_mag_weight=0.5).to(device)

    total_steps = len(dataloader)
    print_freq = max(1, total_steps // 10)

    if rank == 0:
        print("==========================================================", flush=True)
        print(f"🏁 Dual RVQ-GAN V2 (Grid: [16x144]=2304 tokens, Commitment: {commitment_cost}, DiscW: {disc_weight})", flush=True)
        print(f"   • Range Epoche: [{start_epoch} -> {end_epoch - 1}] (Pianificate totali: {total_planned})", flush=True)
        print(f"   • Discriminator Start: Epoca {DISC_START_EPOCH} | Weight: {disc_weight} (Anti-blur enforcement)", flush=True)
        print(f"   • Loss Acustica: MultiScale (L1 + Gradienti Tempo/Freq + Log-Spectral)", flush=True)
        print(f"   • Allineamento Coseno: Frequency-Weighted (Bande Alte x2.5)", flush=True)
        print("==========================================================
", flush=True)

    optimizer_g.zero_grad(set_to_none=True)
    optimizer_d.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, end_epoch):
        if epoch >= split_epoch:
            mel_vae.module.quantizer.set_decay(0.995)
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
            cur_disc_weight = disc_weight * ramp_factor
        else:
            cur_disc_weight = 0.0

        for step, (raw_audio, class_labels) in enumerate(dataloader):
            raw_audio = raw_audio.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)

            if epoch < 15:
                fraction_id = int(np.random.choice([6, 12, 16]))
            elif epoch < 35:
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

                targets_t = class_labels.unsqueeze(1).expand(-1, 144)
                loss_cls_mel = F.cross_entropy(logits_mel_t, targets_t)
                loss_cls_oct = F.cross_entropy(logits_oct_t, targets_t)
                loss_cls = 0.5 * (loss_cls_mel + loss_cls_oct)

                loss_align_cos = compute_frequency_weighted_cosine_loss(z_e_oct, z_e_mel.detach())

                loss_codebook_distill = compute_codebook_distillation_loss(
                    mel_vae.module.quantizer, z_e_oct, z_e_mel, temperature=0.1
                )

                loss_entropy = compute_batch_entropy_loss(
                    mel_vae.module.quantizer, z_e_oct, num_embeddings=num_embeddings, temperature=0.2
                )

                cross_mel_recon = mel_vae.module.decoder(z_q_oct)
                loss_cross = acoustic_loss_fn(cross_mel_recon, x_mel_target)

                loss_g_base = (loss_rec_mel + loss_rec_oct) +                               loss_vq +                               (args.cls_weight * loss_cls) +                               (cos_align_weight * loss_align_cos) +                               (args.codebook_distill_weight * loss_codebook_distill) +                               (args.entropy_weight * loss_entropy) +                               (1.5 * loss_cross)

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

                    loss_g_total = loss_g_base + (FM_WEIGHT * loss_fm_total) + (cur_disc_weight * loss_adv_total)
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
            running_align += (cos_align_weight * loss_align_cos).item()
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

                    m_rec_mel = running_rec_mel / print_freq
                    m_rec_oct = running_rec_oct / print_freq
                    m_cross = running_cross / print_freq
                    m_align = running_align / print_freq
                    m_vq = running_vq / print_freq
                    m_adv = running_adv / print_freq

                    print(f" ⏱️ Ep [{epoch:02d}] {percent:4.1f}% | G: {avg_g:.3f} | "
                          f"[RecM: {m_rec_mel:.3f} | Cross: {m_cross:.3f} | Cos: {m_align:.3f} | VQ: {m_vq:.3f} | Adv: {m_adv:.3f}] | "
                          f"L1: {token_match_l1:4.1f}% | Perpl: {perp:.1f} | ||g_oct||: {norm_g_oct:.2f} | LR: {current_lr:.1e} | {elapsed:.1f}s", flush=True)

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
                    'commitment_cost': commitment_cost,
                    'cos_align_weight': cos_align_weight
                }, checkpoint_path)
                print(f"💾 Checkpoint Dual RVQ-GAN salvato in: {checkpoint_path}", flush=True)

    dataset.close()
    cleanup_distributed_environment(rank)


if __name__ == "__main__":
    main()

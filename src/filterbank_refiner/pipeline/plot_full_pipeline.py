import os
import sys
import argparse
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from utils import get_config_from_yaml
from filterbank_diffusion.data.dataset import DistributedAudioRAWDataset
from filterbank_vae.pipeline.spectral import OnlineSpectrogramPipeline
from filterbank_vae.models.vae import MelVQVAE, OctaveVQVAE
from filterbank_refiner.models.unet_refiner import MelRefinerUNet

LATENT_DIM = int(os.environ.get("LATENT_DIM", 256))
NUM_EMBEDDINGS = int(os.environ.get("NUM_EMBEDDINGS", 1024))
NUM_QUANTIZERS = int(os.environ.get("NUM_QUANTIZERS", 4))


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Full Generative Pipeline (Residual-GAN)")
    parser.add_argument("--vae_checkpoint", type=str, default=os.environ.get("LOCAL_VAE_CHECKPOINT", ""))
    parser.add_argument("--refiner_checkpoint", type=str, default=os.environ.get("LOCAL_REFINER_CHECKPOINT", ""))
    parser.add_argument("--target_fraction", type=int, default=int(os.environ.get("TARGET_FRACTION", "3")))
    parser.add_argument("--samples_per_class", type=int, default=2)
    return parser.parse_args()


def plot_pipeline_diagnostic(pristine_mel, recon_mel_intra, recon_mel_cross_raw, recon_mel_refined,
                             cond_oct, recon_oct_intra, emb_diff_map,
                             class_name, sample_idx, fraction, tok_acc, save_path):
    fig, axes = plt.subplots(7, 1, figsize=(11, 16), sharex=False)

    vmin_mel = float(min(pristine_mel.min(), recon_mel_cross_raw.min(), recon_mel_refined.min(), recon_mel_intra.min()))
    vmax_mel = float(max(pristine_mel.max(), recon_mel_cross_raw.max(), recon_mel_refined.max(), recon_mel_intra.max()))

    vmin_oct = float(min(cond_oct.min(), recon_oct_intra.min()))
    vmax_oct = float(max(cond_oct.max(), recon_oct_intra.max()))

    # 1. Pristine Mel Target
    im0 = axes[0].imshow(pristine_mel, origin='lower', aspect='auto', cmap='magma', vmin=vmin_mel, vmax=vmax_mel)
    axes[0].set_title(f"Class: {class_name} | #{sample_idx} - Pristine Mel ($X_0$)", fontsize=10, fontweight='bold')
    axes[0].set_ylabel("Mel (64)", fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.015, pad=0.01)

    # 2. Intra Mel Autoencoding
    im1 = axes[1].imshow(recon_mel_intra, origin='lower', aspect='auto', cmap='magma', vmin=vmin_mel, vmax=vmax_mel)
    axes[1].set_title("1. Intra Mel Reconstruction (Mel VAE Upper-Bound)", fontsize=10)
    axes[1].set_ylabel("Mel (64)", fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.015, pad=0.01)

    # 3. Cross Mel Grezzo VAE
    im2 = axes[2].imshow(recon_mel_cross_raw, origin='lower', aspect='auto', cmap='magma', vmin=vmin_mel, vmax=vmax_mel)
    axes[2].set_title("2. Stage 1: Cross-Domain Restored Mel (Octave VAE -> Mel Decoder)", fontsize=10)
    axes[2].set_ylabel("Mel (64)", fontsize=8)
    fig.colorbar(im2, ax=axes[2], fraction=0.015, pad=0.01)

    # 4. Refined Mel Residual-GAN
    im3 = axes[3].imshow(recon_mel_refined, origin='lower', aspect='auto', cmap='magma', vmin=vmin_mel, vmax=vmax_mel)
    axes[3].set_title("3. Stage 2: Refined Mel (Residual-GAN 1-Step Super-Resolution)", fontsize=10, fontweight='bold')
    axes[3].set_ylabel("Mel (64)", fontsize=8)
    fig.colorbar(im3, ax=axes[3], fraction=0.015, pad=0.01)

    # 5. Input Ottava Degradata PCHIP
    im4 = axes[4].imshow(cond_oct, origin='lower', aspect='auto', cmap='viridis', vmin=vmin_oct, vmax=vmax_oct)
    axes[4].set_title(f"Input Degraded Filterbank 1/{fraction} Octave (PCHIP 320 Bands)", fontsize=10)
    axes[4].set_ylabel("Octave (320)", fontsize=8)
    fig.colorbar(im4, ax=axes[4], fraction=0.015, pad=0.01)

    # 6. Intra Octave Autoencoding
    im5 = axes[5].imshow(recon_oct_intra, origin='lower', aspect='auto', cmap='viridis', vmin=vmin_oct, vmax=vmax_oct)
    axes[5].set_title("Intra Octave Autoencoding", fontsize=10)
    axes[5].set_ylabel("Octave (320)", fontsize=8)
    fig.colorbar(im5, ax=axes[5], fraction=0.015, pad=0.01)

    # 7. Mappa Differenza L2 Spazio Latente VAE
    im6 = axes[6].imshow(emb_diff_map, origin='lower', aspect='auto', cmap='inferno')
    axes[6].set_title(f"VAE Latent Difference $\|z_q^{{mel}} - z_q^{{oct}}\|_2$ | Token Match: {tok_acc:.1f}%", fontsize=10)
    axes[6].set_xlabel("Latent Time Steps (72)", fontsize=9)
    axes[6].set_ylabel("Freq Lat (4)", fontsize=8)
    fig.colorbar(im6, ax=axes[6], fraction=0.015, pad=0.01)

    plt.tight_layout()
    plt.savefig(save_path, dpi=90, format="jpg", pil_kwargs={"quality": 85})
    plt.close()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes_list, _, _, _, sampling_rate, _, _, _, _, _, _ = get_config_from_yaml("config0.yaml")

    weights_path = os.environ.get("LOCAL_CLAP_WEIGHTS_PATH", ".clap_weights/CLAP_weights_2023.pth")
    spectral_pipeline = OnlineSpectrogramPipeline(weights_path=weights_path, sample_rate=sampling_rate, device=device).to(device)

    mel_vae = MelVQVAE(base_channels=64, latent_dim=LATENT_DIM, num_embeddings=NUM_EMBEDDINGS, num_quantizers=NUM_QUANTIZERS).to(device)
    octave_vae = OctaveVQVAE(base_channels=64, latent_dim=LATENT_DIM, num_embeddings=NUM_EMBEDDINGS, num_quantizers=NUM_QUANTIZERS, shared_quantizer=mel_vae.quantizer).to(device)
    ckpt_vae = torch.load(args.vae_checkpoint, map_location=device)
    mel_vae.load_state_dict({k.replace('module.', ''): v for k, v in ckpt_vae['mel_vae_state_dict'].items()})
    octave_vae.load_state_dict({k.replace('module.', ''): v for k, v in ckpt_vae['octave_vae_state_dict'].items()})
    mel_vae.eval()
    octave_vae.eval()

    refiner = MelRefinerUNet(in_channels=1, out_channels=1, base_channels=32).to(device)
    ckpt_ref = torch.load(args.refiner_checkpoint, map_location=device)
    refiner.load_state_dict({k.replace('module.', ''): v for k, v in ckpt_ref['refiner_state_dict'].items()})
    refiner.eval()

    raw_dataset_root = os.path.join(os.environ.get("BASEDIR", "/tmp"), "dataSEC", "RAW_DATASET", "raw_wav")
    test_dataset = DistributedAudioRAWDataset(base_dir=raw_dataset_root, split="test", target_samples_per_class=args.samples_per_class)

    output_dir = os.path.join(src_root, "results", "full_pipeline_plots", f"fraction_1_{args.target_fraction}")
    os.makedirs(output_dir, exist_ok=True)

    class_collected = {c: [] for c in classes_list}
    for i in range(len(test_dataset)):
        raw_audio, class_idx = test_dataset[i]
        c_name = classes_list[class_idx]
        if len(class_collected[c_name]) < args.samples_per_class:
            class_collected[c_name].append((raw_audio, i))

    with torch.no_grad():
        for c_name, samples in class_collected.items():
            print(f"   ▶ Generazione campioni completi per: {c_name}")
            for s_idx, (audio_tensor, _) in enumerate(samples):
                audio_batch = audio_tensor.unsqueeze(0).to(device)
                x_0_pristine, x_oct_pchip = spectral_pipeline(audio_batch, format_id=1, fraction_id=args.target_fraction, device=device)

                recon_intra, _, z_e_mel, z_q_mel, indices_mel = mel_vae(x_0_pristine)
                recon_oct, _, z_e_oct, z_q_oct, indices_oct = octave_vae(x_oct_pchip)
                recon_cross_raw = mel_vae.decoder(z_q_oct)

                # 1-Step Residual Generation
                recon_refined = refiner(recon_cross_raw)

                l2_diff = torch.norm(z_q_mel - z_q_oct, p=2, dim=1).squeeze(0).cpu().numpy()
                tok_match = float((indices_mel == indices_oct).float().mean().item() * 100.0)

                pristine_np = np.nan_to_num(x_0_pristine.squeeze().cpu().numpy(), nan=0.0).astype(np.float32)
                recon_intra_np = np.nan_to_num(recon_intra.squeeze().cpu().numpy(), nan=0.0).astype(np.float32)
                recon_cross_raw_np = np.nan_to_num(recon_cross_raw.squeeze().cpu().numpy(), nan=0.0).astype(np.float32)
                recon_refined_np = np.nan_to_num(recon_refined.squeeze().cpu().numpy(), nan=0.0).astype(np.float32)
                cond_np = np.nan_to_num(x_oct_pchip.squeeze().cpu().numpy(), nan=0.0).astype(np.float32)
                recon_oct_np = np.nan_to_num(recon_oct.squeeze().cpu().numpy(), nan=0.0).astype(np.float32)

                base_name = f"{c_name}_sample_{s_idx+1}"
                img_path = os.path.join(output_dir, f"{base_name}.jpg")

                plot_pipeline_diagnostic(
                    pristine_np, recon_intra_np, recon_cross_raw_np, recon_refined_np,
                    cond_np, recon_oct_np, l2_diff,
                    c_name, s_idx + 1, args.target_fraction, tok_match, img_path
                )

    test_dataset.close()
    print(f"\n🎉 Tutti i plot a 7 pannelli salvati in: {output_dir}")


if __name__ == "__main__":
    main()

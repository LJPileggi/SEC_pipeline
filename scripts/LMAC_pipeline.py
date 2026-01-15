import os
import sys
import torch
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from tqdm import tqdm
import transformers

# Priorità assoluta moduli locali
sys.path.insert(0, os.getcwd())

from src.models import FinetunedModel, CLAP_initializer
from src.explainability.LMAC import LMAC, Decoder
from src.utils import (
    HDF5EmbeddingDatasetsManager, 
    reconstruct_tracks_from_embeddings, 
    get_config_from_yaml
)
from src.dirs_config import basedir_preprocessed, results_filepath_project, basedir_raw

# 🎯 MONKEY PATCH LOGIC (da get_clap_embeddings.py)
def universal_path_redirect(*args, **kwargs):
    text_path = os.getenv("CLAP_TEXT_ENCODER_PATH")
    if text_path:
        return text_path
    return args[0] if args else None

# Applichiamo la patch prima di qualsiasi inizializzazione di CLAP
transformers.utils.hub.cached_file = universal_path_redirect
transformers.utils.hub.get_checkpoint_shard_files = universal_path_redirect

def parsing():
    parser = argparse.ArgumentParser(description='L-MAC Offline Explainability Pipeline')
    parser.add_argument('--ids_file', type=str, required=True)
    parser.add_argument('--config_file', type=str, default='config0.yaml')
    parser.add_argument('--audio_format', type=str, required=True)
    parser.add_argument('--n_octave', type=int, required=True)
    parser.add_argument('--cut_secs', type=float, required=True)
    parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('--weights_path', type=str, required=True)
    return parser.parse_args()

def save_audio_pydub(waveform, sample_rate, save_path, fmt):
    audio_np = waveform.squeeze().cpu().numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    audio_segment = AudioSegment(
        audio_int16.tobytes(), frame_rate=sample_rate,
        sample_width=2, channels=1
    )
    if fmt == 'mp3':
        audio_segment.export(save_path, format="mp3", bitrate="320k")
    else:
        audio_segment.export(save_path, format=fmt)

def main():
    args = parsing()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Config & Setup
    classes, _, _, _, sampling_rate, _, _, _, _, _, _ = get_config_from_yaml(args.config_file)
    output_base = os.path.join(results_filepath_project, 'explainability', 
                               args.audio_format, f"{args.n_octave}_octave", f"{args.cut_secs}_secs")
    interp_dir = os.path.join(output_base, 'interpretations')
    mask_dir = os.path.join(output_base, 'masks_vis')
    os.makedirs(interp_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # 2. Initialize Models (CLAP ora userà la patch offline)
    print(f"📦 Initializing Models (Offline Mode Patch Active)...")
    classifier = FinetunedModel(classes=classes, device=device, weights_path=args.weights_path)
    clap_model, audio_embedding, _ = CLAP_initializer(device=device, use_cuda=torch.cuda.is_available())
    
    h5_path = os.path.join(basedir_preprocessed, args.audio_format, f"{args.n_octave}_octave", 
                           f"{args.cut_secs}_secs", f"combined_{args.split}.h5")
    
    emb_manager = HDF5EmbeddingDatasetsManager(h5_path, mode='r')
    spec_shape = emb_manager.hf.attrs['spec_shape']
    
    decoder = Decoder(latent_dim_input=1024, output_spectrogram_shape=spec_shape).to(device)
    lmac = LMAC(classifier=classifier, decoder=decoder, clap_model=clap_model, audio_embedding=audio_embedding)

    # 3. Load & Reconstruct
    with open(args.ids_file, 'r') as f:
        target_ids = [line.strip() for line in f if line.strip()]
    
    raw_reconstructed = reconstruct_tracks_from_embeddings(basedir_raw, h5_path, target_ids)
    summary_results = []
    
    # 4. Processing Loop
    for emb_id in tqdm(target_ids, desc="Explainability Analysis"):
        idx_in_h5 = list(emb_manager.existing_keys).index(emb_id)
        data = emb_manager.hf['embedding_dataset'][idx_in_h5]
        
        h_orig = torch.from_numpy(data['embeddings']).float().to(device).unsqueeze(0)
        spec_orig = torch.from_numpy(data['spectrograms']).float().to(device).unsqueeze(0)

        with torch.no_grad():
            mask = lmac.decoder(h_orig)
            listenable_audio = lmac.generate_listenable_interpretation(mask, spec_orig, sampling_rate)
            
            # Fidelity Check
            orig_logits = classifier(h_orig)
            output_masked = audio_embedding(listenable_audio)
            h_masked = output_masked[0] if isinstance(output_masked, (tuple, list)) else output_masked
            masked_logits = classifier(h_masked)

        # Save Outputs
        save_path = os.path.join(interp_dir, f"{emb_id}.{args.audio_format}")
        save_audio_pydub(listenable_audio, sampling_rate, save_path, args.audio_format)
        
        plt.figure(figsize=(10, 4))
        plt.imshow(mask.squeeze().cpu().numpy(), aspect='auto', origin='lower', cmap='magma')
        plt.title(f"L-MAC Mask: {emb_id}")
        plt.savefig(os.path.join(mask_dir, f"{emb_id}_mask.png"))
        plt.close()
        
        summary_results.append({
            "id": emb_id,
            "original_class": classes[torch.argmax(orig_logits).item()],
            "original_conf": torch.softmax(orig_logits, dim=1).max().item(),
            "masked_conf": torch.softmax(masked_logits, dim=1).max().item()
        })

    with open(os.path.join(output_base, 'explainability_summary.json'), 'w') as f:
        json.dump(summary_results, f, indent=4)

    emb_manager.close()
    print(f"🏁 Done! Results: {output_base}")

if __name__ == "__main__":
    main()

import os
import sys
import torch
import json
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import transformers

# Priorità assoluta moduli locali per importare src
sys.path.insert(0, os.getcwd())

from src.models import FinetunedModel, CLAP_initializer
from src.explainability.LMAC import LMAC, Decoder
from src.explainability.xai_utils import save_explanation_audio
from src.utils import (
    HDF5EmbeddingDatasetsManager, 
    reconstruct_tracks_from_embeddings, 
    get_config_from_yaml
)
from src.dirs_config import basedir_preprocessed, results_filepath_project, basedir_raw

# 🎯 MONKEY PATCH LOGIC per ambiente Leonardo
def universal_path_redirect(*args, **kwargs):
    weights_path = os.getenv("LOCAL_CLAP_WEIGHTS_PATH")
    text_path = os.getenv("CLAP_TEXT_ENCODER_PATH")
    if any(x for x in args if 'msclap' in str(x)) or 'CLAP_weights' in str(kwargs):
        return weights_path
    filename = kwargs.get('filename') or (args[1] if len(args) > 1 else None)
    if filename and text_path:
        return os.path.join(text_path, str(filename))
    return text_path

huggingface_hub.hf_hub_download = universal_path_redirect
transformers.utils.hub.cached_file = universal_path_redirect
msclap.CLAPWrapper.hf_hub_download = universal_path_redirect

def parsing():
    parser = argparse.ArgumentParser(description='L-MAC Production Pipeline - Modular Version')
    parser.add_argument('--ids_file', type=str, required=True, help='Path to .txt with embedding IDs')
    parser.add_argument('--config_file', type=str, default='config0.yaml')
    parser.add_argument('--audio_format', type=str, required=True, help='wav, mp3, or flac')
    parser.add_argument('--n_octave', type=int, required=True)
    parser.add_argument('--cut_secs', type=int, required=True) # Forzato int per coerenza path
    parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('--weights_path', type=str, required=True, help='Path to classifier weights')
    return parser.parse_args()

def main():
    args = parsing()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Global Config
    classes, _, _, _, sampling_rate, _, _, _, _, _, _ = get_config_from_yaml(args.config_file)
    
    # 2. Setup Output Directories
    # Usiamo args.cut_secs (int) per evitare il bug del .0 nel path
    output_base = os.path.join(results_filepath_project, 'explainability', 'LMAC',
                               args.audio_format, f"{args.n_octave}_octave", f"{args.cut_secs}_secs")
    interp_dir = os.path.join(output_base, 'interpretations')
    mask_dir = os.path.join(output_base, 'masks_vis')
    os.makedirs(interp_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # 3. Initialize Models
    classifier = FinetunedModel(classes=classes, device=device, weights_path=args.weights_path)
    clap_model, audio_embedding, _ = CLAP_initializer(device=device, use_cuda=torch.cuda.is_available())
    
    # 4. HDF5 and IDs Setup
    h5_path = os.path.join(basedir_preprocessed, args.audio_format, f"{args.n_octave}_octave", 
                           f"{args.cut_secs}_secs", f"combined_{args.split}.h5")
    
    emb_manager = HDF5EmbeddingDatasetsManager(h5_path, mode='r')
    # Conversione shape in tupla di int per compatibilità interpolate
    spec_shape = tuple(int(s) for s in emb_manager.hf.attrs['spec_shape'])
    
    decoder = Decoder(latent_dim_input=1024, output_spectrogram_shape=spec_shape).to(device)
    lmac = LMAC(classifier=classifier, audio_embedding=audio_embedding, decoder=decoder)

    with open(args.ids_file, 'r') as f:
        target_ids = [line.strip() for line in f if line.strip()]
    
    # Reconstruct original audio for the pipeline loop
    raw_reconstructed = reconstruct_tracks_from_embeddings(basedir_raw, h5_path, target_ids)
    
    summary_results = []
    
    # 5. Explainability Loop
    for emb_id in tqdm(target_ids, desc="Processing L-MAC"):
        # Recupero embedding e spettrogramma dall'HDF5
        idx_in_h5 = list(emb_manager.existing_keys).index(emb_id)
        data = emb_manager.hf['embedding_dataset'][idx_in_h5]
        
        h_orig = torch.from_numpy(data['embeddings']).float().to(device).unsqueeze(0)
        spec_linear = torch.from_numpy(data['spectrograms']).float().to(device).unsqueeze(0)

        with torch.no_grad():
            # Generazione Maschera
            mask = lmac.decoder(h_orig)
            
            # Generazione e salvataggio audio interpretazione
            audio_save_path = os.path.join(interp_dir, f"{emb_id}.{args.audio_format}")
            listenable_audio = lmac.generate_listenable_interpretation(
                mask, spec_linear, sampling_rate, 
                save_path=audio_save_path, 
                audio_format=args.audio_format
            )
            
            # Fidelity Analysis (Loopback via CLAP)
            orig_logits = classifier(h_orig)
            output_masked = audio_embedding(listenable_audio)
            h_masked = output_masked[0] if isinstance(output_masked, (tuple, list)) else output_masked
            if h_masked.dim() > 2: h_masked = h_masked[0]
            masked_logits = classifier(h_masked)

        # 6. Save Mask Visualization (.png)
        mask_np = mask.squeeze().cpu().numpy()
        plt.figure(figsize=(10, 4))
        plt.imshow(mask_np, aspect='auto', origin='lower', cmap='magma')
        plt.title(f"L-MAC Mask: {emb_id}")
        plt.colorbar(label="Intensity")
        plt.savefig(os.path.join(mask_dir, f"{emb_id}_mask.png"))
        plt.close()
        
        summary_results.append({
            "id": emb_id,
            "original_class": classes[torch.argmax(orig_logits).item()],
            "original_conf": torch.softmax(orig_logits, dim=1).max().item(),
            "masked_conf": torch.softmax(masked_logits, dim=1).max().item(),
            "audio_path": audio_save_path
        })

    # 7. Final Export
    with open(os.path.join(output_base, 'explainability_summary.json'), 'w') as f:
        json.dump(summary_results, f, indent=4)

    emb_manager.close()
    print(f"✅ L-MAC Completed. Results in: {output_base}")

if __name__ == "__main__":
    main()

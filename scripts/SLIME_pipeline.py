import os
import sys
import gc
import torch
import json
import argparse
import transformers
from tqdm import tqdm

# Priorità assoluta moduli locali
sys.path.insert(0, os.getcwd())

from src.models import FinetunedModel, CLAP_initializer
from src.explainability.SLIME import SLIME
from src.explainability.xai_utils import save_explanation_audio
from src.utils import (
    HDF5EmbeddingDatasetsManager, 
    reconstruct_tracks_from_embeddings, 
    get_config_from_yaml
)
from src.dirs_config import basedir_preprocessed, results_filepath_project, basedir_raw

# 🎯 MONKEY PATCH LOGIC per ambiente Leonardo
def universal_path_redirect(*args, **kwargs):
    text_path = os.getenv("CLAP_TEXT_ENCODER_PATH")
    if text_path:
        return text_path
    return args[0] if args else None

transformers.utils.hub.cached_file = universal_path_redirect
transformers.utils.hub.get_checkpoint_shard_files = universal_path_redirect

def parsing():
    parser = argparse.ArgumentParser(description='SLIME Production Pipeline - Modular Version')
    parser.add_argument('--ids_file', type=str, required=True, help='Path to .txt with embedding IDs')
    parser.add_argument('--config_file', type=str, default='config0.yaml')
    parser.add_argument('--audio_format', type=str, required=True, help='wav, mp3, or flac')
    parser.add_argument('--n_octave', type=int, required=True)
    parser.add_argument('--cut_secs', type=int, required=True) # Coerenza path
    parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('--weights_path', type=str, required=True, help='Path to classifier weights')
    parser.add_argument('--expl_type', type=str, default='time_frequency', choices=['time', 'time_frequency'])
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of perturbations per instance')
    return parser.parse_args()

def main():
    args = parsing()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Configurazione Globale
    classes, _, _, _, sampling_rate, _, _, _, _, _, _ = get_config_from_yaml(args.config_file)
    
    # 2. Setup Directory di Output
    output_base = os.path.join(results_filepath_project, 'explainability', 'SLIME',
                               args.audio_format, f"{args.n_octave}_octave", f"{args.cut_secs}_secs")
    os.makedirs(output_base, exist_ok=True)

    # 3. Inizializzazione Modelli
    classifier = FinetunedModel(classes=classes, device=device, weights_path=args.weights_path)
    clap_model, audio_embedding, _ = CLAP_initializer(device=device, use_cuda=torch.cuda.is_available())
    
    # SLIME inizializzato con il loopback di CLAP
    slime = SLIME(
        classifier=classifier, 
        audio_embedding=audio_embedding, 
        explainer_type=args.expl_type, 
        n_samples=args.n_samples
    )

    # 4. Caricamento Dati HDF5
    h5_path = os.path.join(basedir_preprocessed, args.audio_format, f"{args.n_octave}_octave", 
                           f"{args.cut_secs}_secs", f"combined_{args.split}.h5")
    
    emb_manager = HDF5EmbeddingDatasetsManager(h5_path, mode='r')
    
    with open(args.ids_file, 'r') as f:
        target_ids = [line.strip() for line in f if line.strip()]
    
    # Ricostruzione tracce audio originali per il campionamento SLIME
    raw_tracks = reconstruct_tracks_from_embeddings(basedir_raw, h5_path, target_ids)
    
    summary_results = []
    
    # 5. SLIME Loop
    for emb_id in tqdm(target_ids, desc=f"SLIME Analysis ({args.expl_type})"):
        # Recupero dati dall'HDF5
        idx_in_h5 = list(emb_manager.existing_keys).index(emb_id)
        data = emb_manager.hf['embedding_dataset'][idx_in_h5]
        
        # Spettrogramma lineare e Audio originale
        spec_linear = torch.from_numpy(data['spectrograms']).float().to(device).unsqueeze(0)
        audio_waveform = torch.from_numpy(raw_tracks[emb_id]).float().to(device)
        
        # Determiniamo la classe predetta dal modello per spiegarla
        with torch.no_grad():
            h_orig = torch.from_numpy(data['embeddings']).float().to(device).unsqueeze(0)
            class_idx = torch.argmax(classifier(h_orig)).item()

        # Generazione Spiegazione (Ridge Regression weights)
        explanation = slime.explain_instance(
            audio_waveform=audio_waveform,
            spec_linear=spec_linear,
            sampling_rate=sampling_rate,
            class_idx=class_idx
        )
        
        # Costruiamo un summary coerente
        summary_results.append({
            "id": emb_id,
            "predicted_class": classes[class_idx],
            "explanation": explanation, # Dizionario Componente -> Peso [cite: 176]
            "explainer_type": args.expl_type
        })

        # 🎯 CRITICAL CLEANUP: Libera i grafi di torch di questa iterazione
        del spec_linear
        del audio_waveform
        del h_orig
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # 6. Export Finale
    output_path = os.path.join(output_base, 'slime_summary.json')
    with open(output_path, 'w') as f:
        json.dump(summary_results, f, indent=4)

    emb_manager.close()
    print(f"✅ SLIME Pipeline Completed. Results: {output_path}")

if __name__ == "__main__":
    main()

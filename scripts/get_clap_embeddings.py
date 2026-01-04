import os
import sys

# 🎯 ESSENZIALE: Poiché il codice è dinamico, forziamo Python a usare 
# la cartella montata (/app) come priorità assoluta per src e altri moduli.
sys.path.insert(0, '/app')

import huggingface_hub
import transformers
import msclap

# 🎯 MONKEY PATCH AGGIORNATA: Gestisce redirect di cartelle E file singoli
def universal_path_redirect(*args, **kwargs):
    rank = os.environ.get('SLURM_PROCID', '0')
    
    # 1. Recupero percorsi dalle tue variabili d'ambiente (definite nello .sh)
    weights_path = os.getenv("LOCAL_CLAP_WEIGHTS_PATH")
    text_path = os.getenv("CLAP_TEXT_ENCODER_PATH") # Questo punta a /tmp_data/roberta-base

    # 2. Identificazione del bersaglio della chiamata
    # Spulciamo args e kwargs per capire cosa sta cercando HuggingFace/Transformers
    target_name = ""
    if len(args) > 0: target_name += str(args[0])
    if len(args) > 1: target_name += str(args[1])
    if 'filename' in kwargs: target_name += str(kwargs['filename'])
    if 'pretrained_model_name_or_path' in kwargs: target_name += str(kwargs['pretrained_model_name_or_path'])

    # --- LOGICA DI REDIRECT ---

    # A. Se la chiamata riguarda i pesi CLAP (.pth)
    if 'msclap' in target_name or 'CLAP_weights' in target_name:
        return weights_path

    # B. Se la chiamata riguarda il TextEncoder (RoBERTa / GPT2 fallback)
    # Intercettiamo sia 'roberta' che il percorso fantasma '/opt/models' che causa il NoneType
    if any(x in target_name for x in ['roberta', 'gpt2', '/opt/models', 'config.json', 'pytorch_model.bin']):
        filename = kwargs.get('filename') or (args[1] if len(args) > 1 else None)
        
        if text_path:
            if filename and not os.path.isdir(os.path.join(text_path, str(filename))):
                target = os.path.join(text_path, str(filename))
                # Questo print ti confermerà che stiamo finalmente usando /tmp_data
                print(f"🎯 [Rank {rank}] FORCED REDIRECT: {filename} -> {target}", flush=True)
                return target
            return text_path

    return None

# INIEZIONE TOTALE: Sovrascriviamo ovunque per sicurezza
huggingface_hub.hf_hub_download = universal_path_redirect
transformers.utils.hub.cached_file = universal_path_redirect
transformers.utils.hub.hf_hub_download = universal_path_redirect
msclap.CLAPWrapper.hf_hub_download = universal_path_redirect

print("🚀 [PATCH] Sistema di monitoraggio attivato.", flush=True)

import argparse
import logging
# import torch # Aggiunto per rilevamento GPU locale

sys.path.append('.')

from src.dirs_config import basedir_preprocessed
from src.distributed_clap_embeddings import run_distributed_slurm, run_local_multiprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parsing():
    parser = argparse.ArgumentParser(description='Get CLAP embeddings from audio files')
    parser.add_argument('--config_file', metavar='config_file', dest='config_file',
            help='config file to load to get model and training params.')
    parser.add_argument('--n_octave', metavar='n_octave', dest='n_octave', type=int,
            help='octaveband split for the spectrograms.')
    parser.add_argument('--audio_format', metavar='audio_format', dest='audio_format',
            help='audio format to embed; choose between \'wav\', \'mp3\', \'flac\'.')
    parser.set_defaults(config_file='config0.yaml')
    parser.set_defaults(audio_format='wav')
    args = parser.parse_args()
    return args

def main():
    # 1. Parsing immediato (solo CPU/stringhe)
    args = parsing()
    rank = os.environ.get("SLURM_PROCID", "NON_TROVATO")
    print(f"DEBUG: Rank rilevato dal sistema: {rank}", flush=True)
    
    # 2. Rilevamento ambiente ultra-veloce senza chiamate a torch.cuda
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    
    if slurm_job_id:
        # 🎯 SE SIAMO SU SLURM: non interroghiamo le GPU qui!
        # Lasciamo che sia setup_distributed_environment a farlo dopo
        run_distributed_slurm(args.config_file, args.audio_format, args.n_octave)
    else:
        # Solo in locale inizializziamo CUDA per contare le GPU
        import torch 
        ws = torch.cuda.device_count() if torch.cuda.is_available() else 4
        print(f"Ambiente locale: avvio con {ws} processi...")
        run_local_multiprocess(args.config_file, args.audio_format, args.n_octave, ws)

if __name__ == "__main__":
    main()

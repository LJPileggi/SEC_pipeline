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
    weights_path = os.getenv("LOCAL_CLAP_WEIGHTS_PATH")
    text_path = os.getenv("CLAP_TEXT_ENCODER_PATH") # /tmp_data/roberta-base

    # 1. Intercettazione Pesi CLAP (.pth)
    if any(x for x in args if 'msclap' in str(x)) or 'CLAP_weights' in str(kwargs):
        return weights_path

    # 2. INTERCETTAZIONE TOTALE TRANSFORMERS
    # Qualunque cosa chieda transformers (gpt2, roberta, /opt/models), 
    # noi rispondiamo con il file corrispondente in /tmp_data/roberta-base
    
    # Identifichiamo il file richiesto (config.json, pytorch_model.bin, ecc.)
    filename = kwargs.get('filename') or (args[1] if len(args) > 1 else None)
    
    if filename and text_path:
        # Costruiamo il percorso forzato ignorando l'origine (args[0])
        forced_target = os.path.join(text_path, str(filename))
        
        # Log di guerra per confermare l'esecuzione
        print(f"🎯 [Rank {rank}] FIREWALL REDIRECT: Forzo {filename} -> {forced_target}", flush=True)
        return forced_target

    return text_path

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

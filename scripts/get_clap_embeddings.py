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
    text_path = os.getenv("CLAP_TEXT_ENCODER_PATH") # Punta a /tmp_data/roberta-base

    # 🎯 IDENTIFICAZIONE BERSAGLIO
    arg_str = str(args)
    # Prendiamo il filename se presente in kwargs o args
    filename = kwargs.get('filename') or (args[1] if len(args) > 1 else None)

    # A. CASO PESI CLAP (.pth)
    if 'msclap' in arg_str or 'CLAP_weights' in arg_str:
        return weights_path

    # B. CASO TEXT ENCODER (La trappola per /opt/models)
    # Se il percorso contiene /opt/models o gpt2 o roberta, FORZIAMO il redirect a /tmp_data
    if any(x in arg_str for x in ['/opt/models', 'gpt2', 'roberta']) or \
       (filename and any(f in str(filename) for f in ['config.json', '.bin', '.safetensors', 'vocab', 'merges'])):
        
        if text_path and filename:
            target = os.path.join(text_path, str(filename))
            # 🎯 QUI STA IL TRUCCO: Ignoriamo l'input originale e restituiamo SOLO il nostro path
            print(f"🎯 [Rank {rank}] OVERRIDE AGGRESSIVO: {filename} -> {target}", flush=True)
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

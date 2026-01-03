import os
import huggingface_hub
import transformers

# --- 🎯 MONKEY PATCH NUCLEARE TOTALE (OFFLINE MODE) ---

# 1. Patch per i pesi CLAP (.pth)
def patched_hf_hub_download(*args, **kwargs):
    local_path = os.getenv("LOCAL_CLAP_WEIGHTS_PATH")
    if local_path and os.path.exists(local_path):
        print(f"🎯 [GLOBAL PATCH] Redirect pesi a: {local_path}", flush=True)
        return local_path
    raise FileNotFoundError(f"Pesi non trovati a {local_path}")

# 2. Patch per il TextEncoder (BERT/Tokenizer)
def patched_transformers_cached_file(*args, **kwargs):
    text_path = os.getenv("CLAP_TEXT_ENCODER_PATH")
    if text_path and os.path.exists(text_path):
        # Se transformers cerca un file specifico, lo uniamo al path
        filename = kwargs.get('filename')
        if filename:
            full_path = os.path.join(text_path, filename)
            if os.path.exists(full_path):
                print(f"🎯 [GLOBAL PATCH] Redirect {filename} a: {full_path}", flush=True)
                return full_path
        return text_path
    return None

# Applichiamo le patch globalmente
huggingface_hub.hf_hub_download = patched_hf_hub_download
transformers.utils.hub.cached_file = patched_transformers_cached_file

print("🚀 Patch Offline applicate con successo.", flush=True)

import argparse
import sys
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

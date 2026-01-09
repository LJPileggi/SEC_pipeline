import os
import sys

# 🎯 ESSENZIALE: Priorità assoluta moduli /app
sys.path.insert(0, '/app')

import huggingface_hub
import transformers
import msclap
import argparse
import logging
from src.utils import VERBOSE # Recupero variabile globale
from src.dirs_config import basedir_preprocessed
from src.distributed_clap_embeddings import run_distributed_slurm, run_local_multiprocess

"""
Main entry point for the CLAP embedding pipeline. 
This script handles command-line arguments, detects the execution environment 
(SLURM vs. Local), and applies critical monkey patches to redirect model 
downloads to local pre-cached weights, ensuring operation in firewalled HPC environments.

args:
 - --config_file (str, default: 'config0.yaml'): Name of the YAML config file;
 - --n_octave (int): Resolution of the octave band spectrograms;
 - --audio_format (str, default: 'wav'): Audio extension to process (wav, mp3, flac).

returns:
 - None: Dispatches execution to SLURM or Local orchestrators.
"""

# 🎯 MONKEY PATCH LOGIC
def universal_path_redirect(*args, **kwargs):
    """
    Redirects HuggingFace and MSCLAP download requests to local filesystem paths. 
    This prevents the library from attempting to connect to external hubs, 
    using environment variables to locate local weights and encoders.

    args:
     - *args: Variable length argument list from the original download function;
     - **kwargs: Arbitrary keyword arguments (e.g., 'filename').

    returns:
     - str: The absolute local path to the requested model file or directory.
    """
    rank = os.environ.get('SLURM_PROCID', '0')
    weights_path = os.getenv("LOCAL_CLAP_WEIGHTS_PATH")
    text_path = os.getenv("CLAP_TEXT_ENCODER_PATH")

    # Redirect MSCLAP weights specifically
    if any(x for x in args if 'msclap' in str(x)) or 'CLAP_weights' in str(kwargs):
        return weights_path

    # Redirect general HuggingFace/Transformers files (like RoBERTa config/vocab)
    filename = kwargs.get('filename') or (args[1] if len(args) > 1 else None)
    
    if filename and text_path:
        forced_target = os.path.join(text_path, str(filename))
        # Log patch operations only if diagnostic VERBOSE is active
        if VERBOSE:
            print(f"🎯 [Rank {rank}] FIREWALL REDIRECT: {filename} -> {forced_target}", flush=True)
        return forced_target

    return text_path

# TOTAL INJECTION: Override library methods with our redirector
huggingface_hub.hf_hub_download = universal_path_redirect
transformers.utils.hub.cached_file = universal_path_redirect
transformers.utils.hub.hf_hub_download = universal_path_redirect
msclap.CLAPWrapper.hf_hub_download = universal_path_redirect

def parsing():
    """
    Defines and parses command-line arguments for the embedding pipeline.
    """
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
    """
    Main execution loop. Detects if running under SLURM or locally and 
    triggers the appropriate distributed or multiprocess runner.
    """
    args = parsing()
    
    # 🎯 ENVIRONMENT DETECTION
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    rank = os.environ.get("SLURM_PROCID", "0")
    
    # SETUP SUMMARY: Printed only by Rank 0 to keep logs clean
    if rank == "0" or rank == 0:
        print(f"\n" + "="*50)
        print(f"🎬 STARTING EMBEDDING PIPELINE")
        print(f"   - Config: {args.config_file}")
        print(f"   - Format: {args.audio_format}")
        print(f"   - Octave: {args.n_octave}")
        print(f"   - Slurm Job: {slurm_job_id if slurm_job_id else 'Local'}")
        print("="*50 + "\n", flush=True)
    
    if slurm_job_id:
        # SLURM Mode: leverages srun and process groups
        run_distributed_slurm(args.config_file, args.audio_format, args.n_octave)
    else:
        # Local Mode: calculates available GPUs or defaults to 4 CPU processes
        import torch 
        ws = torch.cuda.device_count() if torch.cuda.is_available() else 4
        if rank == "0" or rank == 0:
            print(f"💻 Local Environment: spawning {ws} processes...", flush=True)
        run_local_multiprocess(args.config_file, args.audio_format, args.n_octave, ws)

if __name__ == "__main__":
    main()

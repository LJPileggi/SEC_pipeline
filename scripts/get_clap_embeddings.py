import argparse
import sys
import os
import logging
import torch # Aggiunto per rilevamento GPU locale

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
    args = parsing()
    
    # Rileva automaticamente quante GPU usare in locale, altrimenti default a 4
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 4

    # Rileva l'ambiente di esecuzione
    if "SLURM_PROCID" in os.environ:
        # In ambiente SLURM, il logging e il setup del rank sono gestiti internamente
        print(f"Ambiente SLURM rilevato. Avvio in modalità distribuita...")
        run_distributed_slurm(args.config_file, args.audio_format, args.n_octave)
    else:
        # Ambiente locale o altro non-SLURM
        print(f"Ambiente locale rilevato. Avvio con {world_size} processi...")
        run_local_multiprocess(args.config_file, args.audio_format, args.n_octave, world_size)

if __name__ == "__main__":
    main()

import argparse
import sys
import os
import logging
sys.path.append('.')

from src.dirs_config import basedir_preprocessed
from src.distributed_clap_embeddings import run_distributed_slurm, run_local_multiprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parsing():
    parser = argparse.ArgumentParser(description='Get CLAP embeddings from audio files')
    parser.add_argument('--config_file', metavar='config_file', dest='config_file',
            help='config file to load to get model and training params.')
    parser.add_argument('--n_octave', metavar='n_octave', dest='n_octave',
            help='octaveband split for the spectrograms.')
    parser.add_argument('--audio_format', metavar='audio_format', dest='audio_format',
            help='audio format to embed; choose between \'wav\', \'mp3\', \'flac\'.')
    parser.set_defaults(config_file='config0.yaml')
    parser.set_defaults(audio_format='wav')
    args = parser.parse_args()
    return args

def main():
    args = parsing()
    world_size = 4
    embed_folder = os.path.join(basedir_preprocessed, f'{args.audio_format}', f'{args.n_octave}_octave')
    if not os.path.exists(embed_folder):
        os.makedirs(embed_folder)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                                             handlers=[logging.StreamHandler(),
                   logging.FileHandler(os.path.join(embed_folder, "log.txt"))])

    # Rileva l'ambiente di esecuzione
    if "SLURM_PROCID" in os.environ:
        print("Ambiente SLURM rilevato. Avvio in modalità distribuita...")
        run_distributed_slurm(args.config_file, args.n_octave, args.audio_format)
    else:
        # Ambiente locale o altro non-SLURM
        print("Ambiente locale rilevato. Avvio in modalità multi-processo...")
        run_local_multiprocess(args.config_file, args.n_octave, args.audio_format, world_size)

if __name__ == "__main__":
    main()

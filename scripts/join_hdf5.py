import argparse
import sys
import os
sys.path.append('.')

from src.utils import get_config_from_yaml, combine_hdf5_files
from src.dirs_config import basedir_preprocessed

def parsing():
    parser = argparse.ArgumentParser(description='Utility to join intermediate hdf5 files.')
    parser.add_argument('--config_file', type=str, default='config0.yaml',
            help='config file to load to get model and training params.')
    parser.add_argument('--n_octave', type=int, required=True,
            help='octaveband split for the spectrograms.')
    parser.add_argument('--audio_format', type=str, default='wav',
            help='audio format to embed; choose between \'wav\', \'mp3\', \'flac\'.')
    return parser.parse_args()

def main():
    args = parsing()
    classes, _, _, _, sample_rate, ref, noise_perc, seed, _, valid_cut_secs, splits_list = get_config_from_yaml(args.config_file)
    octaveband_dir = os.path.join(basedir_preprocessed, f'{args.audio_format}', f'{args.n_octave}_octave')

    combine_hdf5_files(
        root_dir=octaveband_dir, 
        cut_secs_list=valid_cut_secs, 
        embedding_dim=1024, 
        spec_shape=None, # Inferred by the utility
        audio_format=args.audio_format,
        cut_secs=None, # Managed internally by loop
        n_octave=args.n_octave, 
        sample_rate=sample_rate, 
        seed=seed, 
        noise_perc=noise_perc, 
        splits_list=splits_list
    )

if __name__ == "__main__":
    main()

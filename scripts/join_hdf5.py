import argparse
import sys
sys.path.append('.')

from src.utils import get_config_from_yaml, combine_hdf5_files
from src.dirs_config import basedir_preprocessed, basedir_preprocessed_test

def parsing():
    parser = argparse.ArgumentParser(description='Utility to join intermediate hdf5 files.')
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
    _, _, _, _, sample_rate, _, noise_perc, seed, _, valid_cut_secs, splits_list = get_config_from_yaml(config_file)
    octaveband_dir = os.path.join(basedir_preprocessed if args.test_mode=='n' else basedir_preprocessed_test,
                                                                  f'{args.audio_format}', f'{args.n_octave}')
    combine_hdf5_files(octaveband_dir, valid_cut_secs, 1024, None, args.audio_format,
                          valid_cut_secs, args.n_octave, sample_rate, seed, noise_perc, splits_list))

if __name__ == "__main__":
    main()

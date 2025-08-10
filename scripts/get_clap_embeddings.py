import argparse

from .utils import get_config_from_yaml
from src.training import get_embeddings_for_n_octaveband

def parsing():
    parser = argparse.ArgumentParser(description='Get CLAP embeddings from audio files')
    parser.add_argument('--config_file', metavar='config_file', dest='config_file',
            help='config file to load to get model and training params.')
    parser.add_argument('--basedir_raw', metavar='basedir_raw', dest='basedir_raw',
            help='base directory for raw audio; has to match corresponding format.')
    parser.add_argument('--n_octave_dir', metavar='n_octave_dir', dest='n_octave_dir',
            help='embedding directory relative to a certain octave split.')
    parser.add_argument('--audio_format', metavar='audio_format', dest='audio_format',
            help='audio format to embed; choose between \'wav\', \'mp3\', \'flac\'.')
    parser.set_defaults(config_file='config0.yaml')
    parser.set_defaults(audio_format='wav')
    args = parser.parse_args()
    return args

def main():
    args = parsing()
    get_config_from_yaml(args.config_file)
    get_embeddings_for_n_octaveband(args.basedir_raw, args.n_octave_dir, args.audio_format)

if __name__ == "__main__":
    main()

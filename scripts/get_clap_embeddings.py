import argparse

from .utils import get_config_from_yaml
from src.training import get_embeddings_for_n_octaveband

def parsing():
    parser = argparse.ArgumentParser(description='Get CLAP embeddings from audio files')
    parser.add_argument('--config_file', metavar='config_file', dest='config_file',
            help='config file to load to get model and training params.')
    parser.add_argument('--basedir_n_octave', metavar='basedir_n_octave', dest='basedir_n_octave',
            help='base directory relative to a certain octave split.')
    parser.add_argument('--audio_format', metavar='audio_format', dest='audio_format',
            help='audio format to embed; choose between \'wav\', \'mp3\', \'flac\'.')
    parser.set_defaults(config_file='config0.yaml')
    parser.set_defaults(audio_format='wav')
    args = parser.parse_args()
    return args

# TODO: add args.audio_format to get_embeddings_for_n_octaveband
def main():
    args = parsing()
    get_config_from_yaml(args.config_file)
    get_embeddings_for_n_octaveband(args.basedir_n_octave)

if __name__ == "__main__":
    main()

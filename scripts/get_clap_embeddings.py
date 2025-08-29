import argparse
import sys
sys.path.append('../')

from src.utils import get_config_from_yaml
from src.utils_directories import basedir_preprocessed
from src.distributed_clap_embeddings import setup_and_run

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
        os.mkdir(embed_folder)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                                             handlers=[logging.StreamHandler(),
                   logging.FileHandler(os.path.join(embed_folder, "log.txt"))])
    setup_and_run(args.config_file, args.n_octave, args.audio_format, world_size)

if __name__ == "__main__":
    main()

import argparse
import sys
sys.path.append('../')

from src.utils import get_config_from_yaml
from src.utils_directories import *
from src.distributed_clap_embeddings import setup_and_run

def parsing():
    parser = argparse.ArgumentParser(description='Test for CLAP embeddings generation.')
    parser.add_argument('--config_file', metavar='config_file', dest='config_file',
            help='config file to load to get model and training params.')
    parser.set_defaults(config_file='config0.yaml')
    args = parser.parse_args()
    return args

# TODO: expand testing by checking file generation and well functioning of functions
def main():
    args = parsing()
    world_size = 4
    n_octave = 3
    audio_formats = ['wav', 'mp3', 'flac']
    for audio_format in audio_formats:
        print(f'Testing {audio_format} format:')
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                                                handlers=[logging.StreamHandler(),
                                                logging.FileHandler(os.path.join(basedir_preprocessed_test,
                                                    f'{audio_format}', f'{n_octave}_octave'), "log.txt"))])
        setup_and_run(args.config_file, n_octave, audio_format, world_size)

if __name__ == "__main__":
    main()

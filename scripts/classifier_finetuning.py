import argparse

from .utils import get_config_from_yaml
from src.training import get_embeddings_for_n_octaveband

def parsing():
    parser = argparse.ArgumentParser(description='Finetune classifier on CLAP embeddings from audio files')
    parser.add_argument('--config_file', metavar='config_file', dest='config_file',
            help='config file to load to get model and training params.')
    parser.add_argument('--validation_filepath', metavar='validation_filepath', dest='validation_filepath',
            help='directory for validation results.')
    parser.set_defaults(config_file='config0.yaml')
    args = parser.parse_args()
    return args

def main():
    args = parsing()
    get_config_from_yaml(args.config_file)
    select_optim_mainloop(validation_filepath)

if __name__ == "__main__":
    main()

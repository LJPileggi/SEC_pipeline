import argparse

from src.training import get_embeddings_for_n_octaveband

def parsing():
    parser = argparse.ArgumentParser(description='Finetune classifier on CLAP embeddings from audio files')
    parser.add_argument('--validation_filepath', metavar='validation_filepath', dest='validation_filepath',
            help='directory for validation results.')
    args = parser.parse_args()
    return args

# TODO: add args.validation_filepath argument to select_optim_mainloop
def main():
    args = parsing()
    select_optim_mainloop()

if __name__ == "__main__":
    main()

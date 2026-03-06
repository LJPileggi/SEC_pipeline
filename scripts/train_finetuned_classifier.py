import argparse
import os
import sys
sys.path.append('.')
import torch

from src.utils import get_config_from_yaml, load_single_cut_secs_dataloaders
from src.dirs_config import basedir_preprocessed, results_validation_filepath_project
from src.models import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parsing():
    parser = argparse.ArgumentParser(description='Finetune classifier on CLAP embeddings from audio files')
    parser.add_argument('--config_file', metavar='config_file', dest='config_file',
            help='config file to load to get model and training params.')
    parser.add_argument('--n_octave', metavar='n_octave', dest='n_octave',
            help='octaveband split for the spectrograms.')
    parser.add_argument('--audio_format', metavar='audio_format', dest='audio_format',
            help='audio format to embed; choose between \'wav\', \'mp3\', \'flac\'.')
    parser.add_argument('--cut_secs', metavar='cut_secs', dest='cut_secs', type=int,
            help='cut sec to train the classifier on.')
    parser.add_argument('--pretrained_path', metavar='pretrained_path', dest='pretrained_path',
            help='path of pretrained model.')
    parser.set_defaults(config_file='config0.yaml')
    parser.set_defaults(audio_format='wav')
    parser.set_defaults(cut_secs=7)
    parser.set_defaults(pretrained_path=None)
    args = parser.parse_args()
    return args

def main():
    args = parsing()
    # 🎯 FIX: config_file desunto da args
    _, patience, epochs, batch_size, _, _, _, _, _, _, _ = get_config_from_yaml(args.config_file)
    
    octaveband_dir = os.path.join(basedir_preprocessed, args.audio_format, f"{args.n_octave}_octave")
    validation_filepath = os.path.join(results_validation_filepath_project, args.audio_format, args.n_octave)
    os.makedirs(validation_filepath, exist_ok=True)

    optim_config = {
        "optimizer" : {
            "builder" : "Adam",
            "lr" : 0.01
        }
    }

    device = 'cuda' if torch.cuda.is_available() else torch.device('cpu')
    dataloaders, classes = load_single_cut_secs_dataloaders(octaveband_dir, args.cut_secs, 1024, device)
    model = train(dataloaders['train'], dataloaders['es'], optim_config, epochs, patience, device=device,
                                                  classes=classes, pretrained_path=args.pretrained_path)
    FINAL_MODEL_PATH = os.environ.get("FINAL_MODEL_PATH")
    if FINAL_MODEL_PATH:
        torch.save(model.state_dict(), FINAL_MODEL_PATH)

if __name__ == "__main__":
    main()

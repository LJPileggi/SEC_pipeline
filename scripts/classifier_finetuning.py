import argparse
import sys
sys.path.append('.')

from src.utils import get_config_from_yaml
from src.utils_directories import basedir_preprocessed, results_validation_filepath_project
from src.data_handler import load_octaveband_embeddings
from src.models import CLAP_initializer
from src.distributed_finetuning import select_optim_distributed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parsing():
    parser = argparse.ArgumentParser(description='Finetune classifier on CLAP embeddings from audio files')
    parser.add_argument('--config_file', metavar='config_file', dest='config_file',
            help='config file to load to get model and training params.')
    parser.add_argument('--n_octave', metavar='n_octave', dest='n_octave',
            help='octaveband split for the spectrograms.')
    parser.add_argument('--audio_format', metavar='audio_format', dest='audio_format',
            help='audio format to embed; choose between \'wav\', \'mp3\', \'flac\'.')
    parser.add_argument('--model_type', metavar='model_type', dest='model_type',
            help='type of classifier model; choose between \'linear\' and \'xgboost\'.')
    parser.set_defaults(config_file='config0.yaml')
    parser.set_defaults(audio_format='wav')
    parser.set_defaults(model_type='linear')
    args = parser.parse_args()
    return args


def main_worker(rank, world_size, validation_filepath, dataloaders, classes, epochs, patience, clap_model, classifier_model):
    """
    The main function for each process.
    """
    select_optim_distributed(
        rank=rank, 
        world_size=world_size,
        validation_filepath=validation_filepath,
        dataloaders=dataloaders,
        classes=classes,
        epochs=epochs,
        patience=patience,
        clap_model=clap_model,
        classifier_model=classifier_model
    )

def main():
    # Define your parameters here
    args = parsing()
    patience, epochs, batch_size, _, _, _, _, _, _, _, _ = get_config_from_yaml(config_file)
    world_size = 4
    
    # Percorso dove si trovano gli embedding
    octaveband_dir = os.path.join(basedir_preprocessed, f'{args.audio_format}', f'{args.n_octave}')
    validation_filepath = os.path.join(results_validation_filepath_project, f'{args.audio_format}', f'{args.n_octave}')
    if not os.path.exists(validation_filepath):
        os.makedirs(validation_filepath)

    # 1. Carica i dataloader e i dataset
    print("Caricamento degli embeddings in corso...")
    dataloaders_dict, _ = load_octaveband_embeddings(octaveband_dir, batch_size)
    print("Caricamento completato.")
    
    # 2. Ottieni la lista delle classi dal primo dataset caricato
    first_dataset = list(dataloaders_dict.values())[0][0].dataset
    classes = first_dataset.classes
    
    # 3. Inizializza il modello CLAP su CPU, sarà spostato su GPU dai processi
    clap_model, _, _ = CLAP_initializer(device='cpu')

    # 4. Avvia il processo distribuito
    import torch.multiprocessing as mp
    mp.spawn(
        main_worker,
        args=(world_size, validation_filepath, dataloaders_dict, classes, epochs, patience, clap_model, args.classifier_model),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()

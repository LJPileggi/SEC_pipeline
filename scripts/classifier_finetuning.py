import argparse
import sys
sys.path.append('.')

from src.utils import get_config_from_yaml
from src.dirs_config import basedir_preprocessed, results_validation_filepath_project
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

def run_local_worker(rank, world_size, validation_filepath, octaveband_dir, cut_secs_list, epochs, patience, clap_model, model_type):
    """
    Questa funzione ricalca esattamente run_local_multiprocess.
    Il rank viene iniettato automaticamente da mp.spawn come primo argomento.
    """
    # Distribuzione deterministica dei task basata sul rank
    my_cut_secs = cut_secs_list[rank::world_size]
    
    # Chiamata alla logica distribuita
    select_optim_distributed(
        rank=rank, 
        world_size=world_size, 
        validation_filepath=validation_filepath, 
        octaveband_dir=octaveband_dir, 
        my_cut_secs=my_cut_secs, 
        classes_global=None, 
        epochs=epochs, 
        patience=patience, 
        clap_model=clap_model, 
        classifier_model=model_type
    )

def main():
    args = parsing()
    # 🎯 FIX: config_file desunto da args
    _, patience, epochs, batch_size, _, _, _, _, _, cut_secs_list, _ = get_config_from_yaml(args.config_file)
    
    # 🎯 RILEVAMENTO AMBIENTE (Slurm vs Locale)
    is_slurm = os.environ.get('SLURM_JOB_ID') is not None
    if is_slurm:
        rank = int(os.environ.get('SLURM_PROCID', 0))
        world_size = int(os.environ.get('SLURM_NTASKS', 1))
    else:
        rank = 0
        world_size = 1 # In interattivo default 1 processo (o più se mp.spawn)

    octaveband_dir = os.path.join(basedir_preprocessed, args.audio_format, f"{args.n_octave}_octave")
    validation_filepath = os.path.join(results_validation_filepath_project, args.audio_format, args.n_octave)
    os.makedirs(validation_filepath, exist_ok=True)

    # Inizializzazione CLAP (CPU per il passaggio ai worker)
    clap_model, _, _ = CLAP_initializer(device='cpu')

    # 🎯 Distribuzione cut_secs (Logica deterministica)
    my_cut_secs = cut_secs_list[rank::world_size]

    if is_slurm:
        # Modalità SLURM: il processo è già istanziato, chiamiamo direttamente
        select_optim_distributed(rank, world_size, validation_filepath, octaveband_dir, 
                                 my_cut_secs, None, epochs, patience, clap_model, args.model_type)
    else:
        # Modalità Locale: puoi scegliere mp.spawn o chiamata diretta se world_size=1
        if world_size > 1:
            import torch.multiprocessing as mp
            mp.spawn(main_worker_local, nprocs=world_size, args=(world_size, validation_filepath, octaveband_dir, cut_secs_list, epochs, patience, clap_model, args.model_type))
        else:
            select_optim_distributed(0, 1, validation_filepath, octaveband_dir, cut_secs_list, None, epochs, patience, clap_model, args.model_type)


if __name__ == "__main__":
    main()

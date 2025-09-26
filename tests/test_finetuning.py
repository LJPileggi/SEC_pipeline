import sys
import unittest
import shutil
import logging
sys.path.append('.')

from src.utils import get_config_from_yaml, load_octaveband_datasets
from src.dirs_config import basedir_preprocessed, results_validation_filepath_project_test
from src.models import CLAP_initializer
from src.distributed_finetuning import select_optim_distributed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class FinetuningTestCase(unittest.TestCase):
    def setUp(self):
        self._test_config = 'test_config.yaml'
        patience, epochs, batch_size, _, _, _, _, _, _, _, _ = get_config_from_yaml(self._test_config)
        self._n_octave = 3
        self._audio_format = 'wav'
        self._classifier_model = 'linear'
        self._world_size = 4
        octaveband_dir = os.path.join(basedir_preprocessed_test, f'{self._audio_format}', f'{self._n_octave}')
        validation_filepath = os.path.join(results_validation_filepath_project_test, f'{self._audio_format}', f'{self._n_octave}')
        if not os.path.exists(validation_filepath):
            os.makedirs(validation_filepath)
        print("Caricamento degli embeddings in corso...")
        self._dataloaders_dict, _ = load_octaveband_datasets(octaveband_dir, batch_size, ["embeddings"])
        print("Caricamento completato.")

        # 2. Ottieni la lista delle classi dal primo dataset caricato
        first_dataset = list(self._dataloaders_dict.values())[0][0].dataset
        classes = first_dataset.classes

        # 3. Inizializza il modello CLAP su CPU, sarà spostato su GPU dai processi
        clap_model, _, _ = CLAP_initializer(device='cpu')

        # 4. Avvia il processo distribuito
        import torch.multiprocessing as mp
        mp.spawn(
            select_optim_distributed,
            args=(world_size, validation_filepath, self._dataloaders_dict, classes, epochs, patience, clap_model, self._classifier_model),
            nprocs=world_size,
            join=True
        )


    def tearDown(self):
        for splits in self._dataloaders_dict.values():
            for split in splits:
                split.close()

if __name__ == '__main__':
    unittest.main()

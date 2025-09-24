import os
import sys
import unittest
import shutil
import logging
sys.path.append('.')

from src.utils import get_config_from_yaml
from src.dirs_config import *
from src.distributed_clap_embeddings import run_distributed_slurm, run_local_multiprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class EmbeddingTestCase(unittest.TestCase):
    def setUp(self):
        _basedir_raw_format = os.path.join(basedir_raw_test, 'wav')
        _classes_list = sorted([d for d in os.listdir(_basedir_raw_format) if os.path.isdir(os.path.join(_basedir_raw_format, d))])
        self.n_classes = len(_classes_list)
        self._gen_test_embeddings()

    def tearDown(self):
        # delete_files = input("Do you want to delete all embeddings and spectrograms? y/n ")
        # if delete_files.lower() in ["y", "yes"]:
        #     for d in os.listdir(basedir_preprocessed_test):
        #         if os.path.isdir(os.path.join(basedir_preprocessed_test, d)):
        #             shutil.rmtree(os.path.join(basedir_preprocessed_test, d))
        # else:
        #     pass
        pass

    def _gen_test_embeddings(self):
        self.test_config = 'test_config.yaml'
        _world_size = 4
        self._n_octave = 3
        self._audio_formats = ['wav', 'mp3', 'flac']
        _, _, _, _, _, _, _, _, _, valid_cut_secs, divisions_xc_sizes_names = get_config_from_yaml(self.test_config)
        self._valid_cut_secs = valid_cut_secs
        self._divisions_xc_sizes_names = divisions_xc_sizes_names
        for audio_format in self._audio_formats:
            print(f'Testing {audio_format} format:')
            embed_dir = os.path.join(basedir_preprocessed_test, f'{audio_format}', f'{self._n_octave}_octave')
            if not os.path.exists(embed_dir):
                os.makedirs(embed_dir)
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                                                    handlers=[logging.StreamHandler(),
                                                    logging.FileHandler(os.path.join(embed_dir, "log.txt"))])
            # Rileva l'ambiente di esecuzione
            if "SLURM_PROCID" in os.environ:
                print("Ambiente SLURM rilevato. Avvio in modalità distribuita...")
                run_distributed_slurm(self.test_config, audio_format, self._n_octave, test=True)
            else:
                # Ambiente locale o altro non-SLURM
                print("Ambiente locale rilevato. Avvio in modalità multi-processo...")
                run_local_multiprocess(self.test_config, audio_format, self._n_octave, _world_size, test=True)

    def _n_files_within_subfolders(self, extension, audio_format):
        all_files = []
        # 1. Itera su tutte le directory e i file a partire dal percorso specificato
        for dirpath, dirnames, filenames in os.walk(os.path.join(basedir_preprocessed_test, f'{audio_format}', f'{self._n_octave}_octave')):
            # 2. Itera su tutti i file trovati nella directory corrente
            for filename in filenames:
                # 3. Controlla l'estensione del file
                if filename.endswith(f'{extension}'):
                    # 4. Aggiungi il percorso completo del file alla lista
                    full_path = os.path.join(dirpath, filename)
                    all_files.append(full_path)
        return len(all_files)

    def test_number_of_embeddings(self):
        correct_n = self.n_classes * sum(div[1] for div in self._divisions_xc_sizes_names) * len(self._valid_cut_secs)
        for audio_format in self._audio_formats:
            n_segments = self._n_files_within_subfolders(audio_format, audio_format)
            n_spec = self._n_files_within_subfolders('npy', audio_format)
            n_embeddings = self._n_files_within_subfolders('pt', audio_format)
            self.assertEqual(0, n_segments)
            self.assertEqual(correct_n, n_spec)
            self.assertEqual(correct_n, n_embeddings)


if __name__ == '__main__':
    unittest.main()

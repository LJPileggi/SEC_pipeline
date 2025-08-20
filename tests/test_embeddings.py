import argparse
import sys
import unittest
import shutil
sys.path.append('../')

from src.utils import get_config_from_yaml
from src.utils_directories import *
from src.distributed_clap_embeddings import setup_and_run


class EmbeddingTestCase(unittest.TestCase):
    def setUp(self):
        _basedir_raw_format = os.path.join(basedir_raw_test, 'wav')
        _classes_list = sorted([d for d in os.listdir(_basedir_raw_format) if os.path.isdir(os.path.join(_basedir_raw_format, d))])
        self.n_classes = len(_classes_list)
        self._gen_test_embeddings()

    def tearDown(self):
        for d in os.listdir(basedir_preprocessed_test) if os.path.isdir(os.path.join(basedir_preprocessed_test, d)):
            shutil.rmtree(d)

    def _gen_test_embeddings(self):
        self.test_config = os.path.join('..', 'configs', 'test_config.yaml')
        _world_size = 4
        self._n_octave = 3
        self._audio_formats = ['wav', 'mp3', 'flac']
        for audio_format in self._audio_formats:
            print(f'Testing {audio_format} format:')
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                                                    handlers=[logging.StreamHandler(),
                                                    logging.FileHandler(os.path.join(basedir_preprocessed_test,
                                                    f'{audio_format}', f'{self._n_octave}_octave'), "log.txt"))])
            setup_and_run(self.test_config, n_octave, audio_format, _world_size)

    def test_number_of_embeddings(self):
        correct_n = self.n_classes * sum(div[1] for div in divisions_xc_sizes_names)
        for audio_format in self._audio_formats:
            n_segments = sum(len(files) for _, _, files in os.walk(os.path.join(basedir_preprocessed_test,
                                f'{audio_format}', f'{self._n_octave}_octave')) if files.find(f'.{audio_format}') != -1)
            n_spec = sum(len(files) for _, _, files in os.walk(os.path.join(basedir_preprocessed_test,
                                f'{audio_format}', f'{self._n_octave}_octave')) if files.find('.npy') != -1)
            n_embeddings = sum(len(files) for _, _, files in os.walk(os.path.join(basedir_preprocessed_test,
                                f'{audio_format}', f'{self._n_octave}_octave')) if files.find('.pt') != -1)
            self.assertEqual(correct_n, n_segments)
            self.assertEqual(correct_n, n_spec)
            self.assertEqual(correct_n, n_embeddings)


if __name__ == '__main__':
    unittest.main()

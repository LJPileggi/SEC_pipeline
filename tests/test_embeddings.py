import argparse
import os
import sys
import unittest
import shutil
import logging
sys.path.append('.')

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
        # for d in os.listdir(basedir_preprocessed_test):
        #     if os.path.isdir(os.path.join(basedir_preprocessed_test, d)):
        #         shutil.rmtree(d)
        pass

    def _gen_test_embeddings(self):
        self.test_config = 'test_config.yaml'
        _world_size = 4
        self._n_octave = 3
        self._audio_formats = ['wav', 'mp3', 'flac']
        _, _, _, _, _, _, _, _, _, _, divisions_xc_sizes_names = get_config_from_yaml(self.test_config)
        self._divisions_xc_sizes_names = divisions_xc_sizes_names
        for audio_format in self._audio_formats:
            print(f'Testing {audio_format} format:')
            embed_dir = os.path.join(basedir_preprocessed_test, f'{audio_format}', f'{self._n_octave}_octave')
            if not os.path.exists(embed_dir):
                os.makedirs(embed_dir)
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                                                    handlers=[logging.StreamHandler(),
                                                    logging.FileHandler(os.path.join(embed_dir, "log.txt"))])
            setup_and_run(self.test_config, audio_format, self._n_octave, _world_size, test=True)

    def test_number_of_embeddings(self):
        correct_n = self.n_classes * sum(div[1] for div in self._divisions_xc_sizes_names)
        for audio_format in self._audio_formats:
            n_segments = sum(len([f for f in files if f.find(f'.{audio_format}') != -1]) \
                      for _, _, files in os.walk(os.path.join(basedir_preprocessed_test,
                                        f'{audio_format}', f'{self._n_octave}_octave')))
            n_spec = sum(len([f for f in files if f.find('.npy') != -1]) for _, _, files \
                                      in os.walk(os.path.join(basedir_preprocessed_test,
                                        f'{audio_format}', f'{self._n_octave}_octave')))
            n_embeddings = sum(len([f for f in files if f.find('.pt') != -1]) for _, _, files \
                                      in os.walk(os.path.join(basedir_preprocessed_test,
                                        f'{audio_format}', f'{self._n_octave}_octave')))
            self.assertEqual(correct_n, n_segments)
            self.assertEqual(correct_n, n_spec)
            self.assertEqual(correct_n, n_embeddings)


if __name__ == '__main__':
    unittest.main()

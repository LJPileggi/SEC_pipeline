import argparse
import sys
import unittest
sys.path.append('../')

from src.utils import get_config_from_yaml
from src.utils_directories import *
from src.distributed_clap_embeddings import setup_and_run


class EmbeddingTestCase(unittest.TestCase):
    def setUp(self):
        self._gen_test_embeddings()

    def tearDown(self):
        pass

    def _gen_test_embeddings(self):
        self.test_config = os.path.join('..', 'configs', 'test_config.yaml')
        world_size = 4
        n_octave = 3
        audio_formats = ['wav', 'mp3', 'flac']
        for audio_format in audio_formats:
            print(f'Testing {audio_format} format:')
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                                                    handlers=[logging.StreamHandler(),
                                                    logging.FileHandler(os.path.join(basedir_preprocessed_test,
                                                    f'{audio_format}', f'{n_octave}_octave'), "log.txt"))])
            setup_and_run(self.test_config, n_octave, audio_format, world_size)

    def test_number_of_embeddings(self):
        pass

# TODO: complete testing

if __name__ == '__main__':
    unittest.main()

import unittest
import os
import shutil
import tempfile
import sys
import torch
import torch.multiprocessing as mp
import numpy as np
import h5py
import pandas as pd
import json
from unittest.mock import patch, MagicMock

# Aggiungi il percorso 'src' al path di sistema per le importazioni relative
# Assumendo che il file di test sia nella root o in una cartella 'tests'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Crea directory di test
BASEDIR_TEST = os.path.join('..', 'tests')
BASEDIR_RAW_TEST = os.path.join(BASEDIR_TEST, 'RAW_DATASET')
BASEDIR_PREPROCESSED_TEST = os.path.join(BASEDIR_TEST, 'PREPROCESSED_DATASET')

os.makedirs(BASEDIR_TEST, exist_ok=True)
os.makedirs(BASEDIR_RAW_TEST, exist_ok=True)
os.makedirs(BASEDIR_PREPROCESSED_TEST, exist_ok=True)

# --- Configurazione Test ---
TEST_CLASSES = ['Music', 'Voices', 'Birds']
TEST_TRACKS_PER_CLASS = 15 # Abbastanza per uno split teorico
TEST_AUDIO_FORMAT = 'wav'
TEST_N_OCTAVE = 2
TEST_CONFIG_FILENAME = 'test_config.yaml' # Nome fittizio del config


# ==============================================================================
# 1. FUNZIONI DI MOCKING E UTILITY DI TEST
# ==============================================================================

# Variabile globale per il DataFrame simulato
MOCKED_ALL_FILES_DF = None

def create_temp_raw_hdf5_dataset(base_raw_dir, classes, tracks_per_class, audio_format, sr=52100):
    """
    Crea una struttura di directory e file HDF5 fittizi che simulano i dati audio grezzi.
    """
    raw_dir = base_raw_dir
    os.makedirs(raw_dir, exist_ok=True)
    all_tracks = []
    
    for class_name in classes:
        class_h5_file = os.path.join(raw_dir, f'{class_name}_raw.h5')
        
        with h5py.File(class_h5_file, 'w') as hf:
            track_name_dset = hf.create_dataset('track_names', (tracks_per_class,), dtype=h5py.string_dtype())
            dt = h5py.vlen_dtype(np.float32)
            audio_data_dset = hf.create_dataset('audio_data', (tracks_per_class,), dtype=dt)

            for i in range(tracks_per_class):
                track_name = f'track_{i:02d}_{class_name}.{audio_format}'
                audio_len = np.random.randint(sr, 3 * sr) # Lunghezza audio casuale
                audio_array = np.random.rand(audio_len).astype(np.float32) * 0.5 

                track_name_dset[i] = track_name.encode('utf-8')
                audio_data_dset[i] = audio_array
                
                all_tracks.append({'track_name': track_name, 'class': class_name, 'filepath': class_h5_file, 'raw_index': i})
                
    return pd.DataFrame(all_tracks)

def mock_get_config_from_yaml_data(config_file):
    """Restituisce i dati di configurazione mockati basati su test_config.yaml."""
    configs = {
        "classes": TEST_CLASSES,
        "patience": 10, "epochs": 10, "batch_size": 2, "save_log_every": 1,
        "sampling_rate": 52100, "ref": 2.e-05, "noise_perc": 0.3, "seed": 1, 
        "center_freqs": [100.0, 500.0],
        "valid_cut_secs": [1.0], # Un solo cut_secs per test più rapidi
        "test_cut_secs": [3.0], # Un altro cut_secs per i test
    }
    # L'output di get_config_from_yaml in utils è una tupla:
    return (configs['patience'], configs['epochs'], configs['batch_size'], configs['save_log_every'],
            configs['sampling_rate'], configs['ref'], configs['noise_perc'], configs['seed'],
            np.array(configs['center_freqs']), configs['valid_cut_secs'], configs['test_cut_secs'],
            configs['classes'])

# Mock per le funzioni distribuite (previene l'uso di risorse reali)
def mock_setup_distributed_environment(rank, world_size, slurm=True):
    # Simula il device, essenziale per la logica successiva
    return torch.device('cuda:0' if slurm else 'cpu') 
    
def mock_cleanup_distributed_environment():
    pass

# Mock per la funzione 'extract_all_files_from_dir' in utils
def mock_extract_all_files_from_dir(base_raw_dir, audio_format):
    global MOCKED_ALL_FILES_DF
    return MOCKED_ALL_FILES_DF

# Mock per la funzione di lavoro principale (process_class_with_cut_secs)
def mock_process_class_with_cut_secs(clap_model, audio_embedding, class_to_process, cut_secs, n_octave, config):
    """Simula il lavoro di embedding, creando i file HDF5 e i log finali di completamento."""
    
    # Simula il percorso di output (usa basedir_preprocessed_test per i test)
    output_dir = os.path.join(BASEDIR_PREPROCESSED_TEST, TEST_AUDIO_FORMAT, f'{n_octave}_octave', f'cut_{cut_secs}')
    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, f'{class_to_process}_emb.h5')
    
    # 1. Crea un file HDF5 fittizio di embedding
    with h5py.File(h5_path, 'w') as hf:
        # Crea dati fittizi di dimensione 1024
        hf.create_dataset('embeddings', data=np.random.rand(TEST_TRACKS_PER_CLASS, 1024).astype(np.float32))
        
    # 2. Simula la creazione del file di log (necessario per la logica di ripresa)
    log_file = os.path.join(output_dir, f'log_{class_to_process}_{cut_secs}.json')
    with open(log_file, 'w') as f:
         json.dump({"completed": True}, f)
         
    return h5_path


# ==============================================================================
# 2. CLASSE DI TEST PRINCIPALE
# ==============================================================================

class TestDistributedClapEmbeddings(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Prepara l'ambiente di test (cartelle e dataset fittizio) prima di tutti i test."""
        
        # 1. Configura la directory temporanea principale
        cls.temp_dir_obj = tempfile.TemporaryDirectory()
        cls.test_root_dir = cls.temp_dir_obj.name 
        
        # Sovrascrive la variabile d'ambiente usata da dirs_config.py
        os.environ['NODE_TEMP_BASE_DIR'] = os.path.join(cls.test_root_dir, 'dataSEC')
        
        # Re-importa dirs_config per aggiornare i path
        import importlib
        from src import dirs_config, utils
        importlib.reload(dirs_config)
        
        # 2. Imposta i percorsi per il test
        cls.raw_dir = BASEDIR_RAW_TEST
        cls.preprocessed_dir = BASEDIR_PREPROCESSED_TEST
        
        # 3. Crea il dataset HDF5 'RAW' fittizio
        global MOCKED_ALL_FILES_DF
        MOCKED_ALL_FILES_DF = create_temp_raw_hdf5_dataset(cls.raw_dir, TEST_CLASSES, TEST_TRACKS_PER_CLASS, TEST_AUDIO_FORMAT)
        
        # 4. Patcha get_config_from_yaml e extract_all_files_from_dir
        utils.get_config_from_yaml = mock_get_config_from_yaml_data
        utils.extract_all_files_from_dir = mock_extract_all_files_from_dir

    @classmethod
    def tearDownClass(cls):
        """Pulisce l'ambiente di test."""
        # Rimuove la directory temporanea e tutto il suo contenuto
        cls.temp_dir_obj.cleanup()
        # Ripristina la variabile d'ambiente
        del os.environ['NODE_TEMP_BASE_DIR']
    
    def setUp(self):
        # Assicurati che le cartelle di output siano pulite prima di ogni test
        shutil.rmtree(self.preprocessed_dir, ignore_errors=True)
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        # Rimuove le variabili d'ambiente SLURM per non influenzare run_local_multiprocess
        os.environ.pop('SLURM_PROCID', None)
        os.environ.pop('SLURM_NTASKS', None)

    # ==============================================================================
    # 3. TEST FUNZIONALI
    # ==============================================================================

    @patch('src.distributed_clap_embeddings.CLAP_initializer', MagicMock())
    @patch('src.distributed_clap_embeddings.process_class_with_cut_secs', side_effect=mock_process_class_with_cut_secs)
    @patch('src.utils.setup_distributed_environment', side_effect=mock_setup_distributed_environment)
    @patch('src.utils.cleanup_distributed_environment', side_effect=mock_cleanup_distributed_environment)
    def test_run_local_multiprocess_cpu_mode(self, mock_cleanup, mock_setup, mock_process, mock_clap_init):
        """
        Testa il loop principale in modalità multiprocessing locale (CPU).
        Verifica la corretta distribuzione e l'esecuzione di tutti i task.
        """
        
        # Simula world_size=4 (il valore predefinito)
        # Task totali: 3 Classi * 2 cut_secs (1.0 e 3.0) = 6 task
        
        # 1. Esecuzione iniziale
        run_local_multiprocess(TEST_CONFIG_FILENAME, TEST_N_OCTAVE, TEST_AUDIO_FORMAT, world_size=4, test=True)
        
        # Poiché local_worker_process chiama mock_process_class_with_cut_secs, 
        # il conteggio finale di mock_process dovrebbe essere 6 (1 chiamata per task)
        self.assertEqual(mock_process.call_count, 6, "Dovrebbero essere stati eseguiti 6 task totali.")
        
        # 2. Verifica l'esistenza di tutti i 6 file di output
        expected_tasks = [ (1.0, cls) for cls in TEST_CLASSES ] + [ (3.0, cls) for cls in TEST_CLASSES ]
        for cut_secs, class_name in expected_tasks:
            h5_path = os.path.join(self.preprocessed_dir, TEST_AUDIO_FORMAT, f'{TEST_N_OCTAVE}_octave', f'cut_{cut_secs}', f'{class_name}_emb.h5')
            self.assertTrue(os.path.exists(h5_path), f"File HDF5 mancante: {class_name}, cut={cut_secs}")

        # 3. Test di ripresa: Cancella il log di un task e riesegui
        mock_process.reset_mock() # Resetta il contatore delle chiamate
        log_dir_to_delete = os.path.join(self.preprocessed_dir, TEST_AUDIO_FORMAT, f'{TEST_N_OCTAVE}_octave', f'cut_{3.0}')
        log_file_to_delete = os.path.join(log_dir_to_delete, f'log_Birds_3.0.json')
        os.remove(log_file_to_delete)
        
        # Rilancia l'esecuzione
        run_local_multiprocess(TEST_CONFIG_FILENAME, TEST_N_OCTAVE, TEST_AUDIO_FORMAT, world_size=4, test=True)
        
        # 4. Verifica che solo il task mancante sia stato rieseguito
        self.assertEqual(mock_process.call_count, 1, "Solo il task mancante (3.0, Birds) dovrebbe essere stato eseguito.")

    @patch('src.distributed_clap_embeddings.CLAP_initializer', MagicMock())
    @patch('src.distributed_clap_embeddings.process_class_with_cut_secs', side_effect=mock_process_class_with_cut_secs)
    @patch('src.utils.setup_distributed_environment', side_effect=mock_setup_distributed_environment)
    @patch('src.utils.cleanup_distributed_environment', side_effect=mock_cleanup_distributed_environment)
    @patch('os.environ', {'SLURM_PROCID': '0', 'SLURM_NTASKS': '2', 'MASTER_ADDR': 'localhost', 'MASTER_PORT': '29500'}) # Simula Rank 0/2
    def test_run_distributed_slurm_gpu_mode_rank0(self, mock_cleanup, mock_setup, mock_process, mock_clap_init):
        """
        Testa il loop principale in modalità SLURM (GPU distribuita) sul Rank 0.
        Verifica che vengano eseguiti solo i task assegnati.
        """
        
        # Forziamo le variabili d'ambiente per il Rank 0
        os.environ['SLURM_PROCID'] = '0' 
        os.environ['SLURM_NTASKS'] = '2'
        
        # Esecuzione (la funzione simula l'esecuzione del singolo processo Slurm)
        run_distributed_slurm(TEST_CONFIG_FILENAME, TEST_N_OCTAVE, TEST_AUDIO_FORMAT, test=True)
        
        # Task totali: 6. World Size: 2. Rank 0 esegue i task con indice 0, 2, 4 (3 task)
        # Task list: (1.0, Music), (1.0, Voices), (1.0, Birds), (3.0, Music), (3.0, Voices), (3.0, Birds)
        # Rank 0 esegue: (1.0, Music), (1.0, Birds), (3.0, Voices)
        
        # 1. Verifica che siano stati eseguiti 3 task
        self.assertEqual(mock_process.call_count, 3, "Il Rank 0 dovrebbe eseguire i suoi 3 task assegnati.")
        
        # 2. Verifica l'esistenza dei file di output per i task del rank 0
        rank0_tasks = [(1.0, 'Music'), (1.0, 'Birds'), (3.0, 'Voices')]
        for cut_secs, class_name in rank0_tasks:
            h5_path = os.path.join(self.preprocessed_dir, TEST_AUDIO_FORMAT, f'{TEST_N_OCTAVE}_octave', f'cut_{cut_secs}', f'{class_name}_emb.h5')
            self.assertTrue(os.path.exists(h5_path), f"File HDF5 mancante per task: {class_name}, cut={cut_secs} (Rank 0)")
            
        # 3. Verifica l'assenza (simulata) dei file non assegnati
        rank1_tasks = [(1.0, 'Voices'), (3.0, 'Music'), (3.0, 'Birds')]
        for cut_secs, class_name in rank1_tasks:
             h5_path = os.path.join(self.preprocessed_dir, TEST_AUDIO_FORMAT, f'{TEST_N_OCTAVE}_octave', f'cut_{cut_secs}', f'{class_name}_emb.h5')
             # Poiché il mock_process_class_with_cut_secs non è stato chiamato, il file NON dovrebbe esistere
             self.assertFalse(os.path.exists(h5_path), f"File HDF5 ECCESSIVO: {class_name}, cut={cut_secs} (non assegnato al Rank 0)")


if __name__ == '__main__':
    # Esempio di esecuzione: python test_distributed_embeddings.py
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

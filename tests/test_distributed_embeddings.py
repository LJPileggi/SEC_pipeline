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
sys.path.append('.')

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
        
        # 2. Imposta i percorsi per il test (ORA SONO DEFINITI NEL PERCORSO TEMPORANEO)
        # NOTA: Usiamo la directory temporanea come root per il test
        # Definiamo le variabili globali (se usate al di fuori della classe)
        global BASEDIR_RAW_TEST
        global BASEDIR_PREPROCESSED_TEST
        
        BASEDIR_RAW_TEST = os.path.join(cls.test_root_dir, 'RAW_DATASET')
        BASEDIR_PREPROCESSED_TEST = os.path.join(cls.test_root_dir, 'PREPROCESSED_DATASET')

        # 2b. Crea le directory (ORA FUNZIONA PERCHÉ SONO IN cls.test_root_dir)
        os.makedirs(BASEDIR_RAW_TEST, exist_ok=True)
        os.makedirs(BASEDIR_PREPROCESSED_TEST, exist_ok=True)
        
        # 3. Imposta i percorsi come attributi di classe per l'uso nei test
        cls.raw_dir = BASEDIR_RAW_TEST
        cls.preprocessed_dir = BASEDIR_PREPROCESSED_TEST
        
        # 4. Crea il dataset HDF5 'RAW' fittizio
        global MOCKED_ALL_FILES_DF
        MOCKED_ALL_FILES_DF = create_temp_raw_hdf5_dataset(cls.raw_dir, TEST_CLASSES, TEST_TRACKS_PER_CLASS, TEST_AUDIO_FORMAT)
        
        # 5. Patcha get_config_from_yaml e extract_all_files_from_dir
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

    def test_run_local_multiprocess_cpu_mode(self):
        """Testa il loop principale in modalità multiprocessing locale (CPU)."""
        
        # DEFINIZIONE E SETUP DEI MOCK
        mock_worker_process = MagicMock()
        mock_combine_files = MagicMock()
        mock_mp_manager = MagicMock()
        mock_mp_queue = MagicMock()
        
        # Definisci il side effect per simulare la creazione del file di output (necessario per gli asserts di esistenza)
        def mock_worker_process_side_effect(*args, **kwargs):
            # args: (audio_format, n_octave, config, rank, world_size, my_tasks, pbar)
            tasks = args[5] # my_tasks
            for cut_secs, class_name in tasks:
                # Crea un file HDF5 fittizio di output per simulare il lavoro del worker
                h5_path = os.path.join(self.preprocessed_dir, TEST_AUDIO_FORMAT, f'{TEST_N_OCTAVE}_octave',
                                                                  f'cut_{cut_secs}', f'{class_name}_emb.h5')
                os.makedirs(os.path.dirname(h5_path), exist_ok=True)
                with open(h5_path, 'w') as f:
                    f.write("mock content") # Simula l'esistenza del file
        
        mock_worker_process.side_effect = mock_worker_process_side_effect
        
        # Simula il processo mp.Process in modo che chiami il worker mockato
        mock_process = MagicMock()
        
        # AGGIUNGIAMO TUTTI I MOCK IN UN UNICO CONTESTO 'WITH PATCH'
        with patch('src.models.CLAP_initializer', MagicMock()) as mock_clap_init, \
             patch('src.distributed_clap_embeddings.local_worker_process', mock_worker_process), \
             patch('src.utils.combine_hdf5_files', mock_combine_files), \
             patch('src.distributed_clap_embeddings.mp.Manager', return_value=mock_mp_manager), \
             patch.object(mock_mp_manager, 'Queue', return_value=mock_mp_queue), \
             patch('src.distributed_clap_embeddings.mp.Process', return_value=mock_process), \
             patch('src.distributed_clap_embeddings.logging.info', MagicMock()), \
             patch('src.distributed_clap_embeddings.logging.error', MagicMock()), \
             patch('src.utils.get_config_from_yaml', mock_get_config_from_yaml_data), \
             patch('src.dirs_config.LOGS_DIR', self.test_root_dir):
            
            # ESECUZIONE DEL TEST
            run_distributed_embeddings(
                config_file=TEST_CONFIG_FILENAME,
                raw_dir=self.raw_dir,
                preprocessed_dir=self.preprocessed_dir,
                n_octave=TEST_N_OCTAVE,
                slurm=False # Modalità locale
            )
            
            # ASSERTIONS
            self.assertEqual(mock_clap_init.call_count, 1, "CLAP_initializer deve essere chiamato una sola volta nel processo principale.")
            self.assertEqual(mock_process.call_count, 4, "Devono essere avviati 4 processi worker (world_size=4).")
            # Verifica che il combine_hdf5_files sia chiamato una volta dopo il join dei processi
            self.assertEqual(mock_combine_files.call_count, 1, "La combinazione deve avvenire una sola volta.")
            
            # Verifica l'esistenza di tutti i 6 file di output (3 classi * 2 cut_secs)
            total_tasks = [(1.0, c) for c in TEST_CLASSES] + [(3.0, c) for c in TEST_CLASSES]
            for cut_secs, class_name in total_tasks:
                h5_path = os.path.join(self.preprocessed_dir, TEST_AUDIO_FORMAT, f'{TEST_N_OCTAVE}_octave',
                                                                  f'cut_{cut_secs}', f'{class_name}_emb.h5')
                self.assertTrue(os.path.exists(h5_path), f"File HDF5 mancante per task: {class_name}, cut={cut_secs}")

    def test_run_distributed_slurm_gpu_mode_rank0(self):
        """Testa il loop principale in modalità SLURM (GPU distribuita) sul Rank 0."""
        
        # DEFINIZIONE E SETUP DEI MOCK
        mock_process_class = MagicMock()
        mock_combine_files = MagicMock()
        # setup_distributed_environment ritorna: device='cuda', rank=0, world_size=2
        mock_setup_dist = MagicMock(return_value=('cuda', 0, 2)) 
        mock_cleanup_dist = MagicMock()
        mock_log_info = MagicMock()

        # Definisci il side effect per simulare la creazione del file di output
        def mock_process_class_side_effect(*args, **kwargs):
            # args: (clap_model, audio_embedding, class_to_process, cut_secs, n_octave, config)
            # Simula la creazione del file solo per i task eseguiti
            class_name = args[2] 
            cut_secs = args[3] 
            
            # Questo rank (0) dovrebbe processare: (1.0, Music), (1.0, Birds), (3.0, Voices)
            if (cut_secs, class_name) in [(1.0, 'Music'), (1.0, 'Birds'), (3.0, 'Voices')]:
                h5_path = os.path.join(self.preprocessed_dir, TEST_AUDIO_FORMAT,f'{TEST_N_OCTAVE}_octave',
                                                                f'cut_{cut_secs}', f'{class_name}_emb.h5')
                os.makedirs(os.path.dirname(h5_path), exist_ok=True)
                with open(h5_path, 'w') as f:
                    f.write("mock content") # Simula l'esistenza del file
        
        mock_process_class.side_effect = mock_process_class_side_effect
        
        # AGGIUNGIAMO TUTTI I MOCK IN UN UNICO CONTESTO 'WITH PATCH'
        with patch('src.models.CLAP_initializer', MagicMock()) as mock_clap_init, \
             patch('src.distributed_clap_embeddings.process_class_with_cut_secs', mock_process_class), \
             patch('src.utils.combine_hdf5_files', mock_combine_files), \
             patch('src.utils.setup_distributed_environment', mock_setup_dist), \
             patch('src.utils.cleanup_distributed_environment', mock_cleanup_dist), \
             patch('src.distributed_clap_embeddings.logging.info', mock_log_info), \
             patch('src.distributed_clap_embeddings.logging.error', MagicMock()), \
             patch('src.utils.get_config_from_yaml', mock_get_config_from_yaml_data), \
             patch('src.dirs_config.LOGS_DIR', self.test_root_dir):
            
            # ESECUZIONE DEL TEST
            run_distributed_embeddings(
                config_file=TEST_CONFIG_FILENAME,
                raw_dir=self.raw_dir,
                preprocessed_dir=self.preprocessed_dir,
                n_octave=TEST_N_OCTAVE,
                slurm=True
            )
            
            # ASSERTIONS
            self.assertEqual(mock_clap_init.call_count, 1, "CLAP_initializer deve essere chiamato una sola volta.")
            self.assertEqual(mock_process_class.call_count, 3, "Il Rank 0 deve eseguire 3 task.")
            self.assertEqual(mock_combine_files.call_count, 2, "La combinazione deve avvenire per i 2 cut_secs (1.0 e 3.0).")
            
            # Verifica l'esistenza dei file di output per i task del rank 0
            rank0_tasks = [(1.0, 'Music'), (1.0, 'Birds'), (3.0, 'Voices')]
            for cut_secs, class_name in rank0_tasks:
                h5_path = os.path.join(self.preprocessed_dir, TEST_AUDIO_FORMAT,f'{TEST_N_OCTAVE}_octave',
                                                                f'cut_{cut_secs}', f'{class_name}_emb.h5')
                self.assertTrue(os.path.exists(h5_path), f"File HDF5 mancante per task: {class_name}, cut={cut_secs} (Rank 0)")
                
            mock_cleanup_dist.assert_called_once()


if __name__ == '__main__':
    # Esempio di esecuzione: python test_distributed_embeddings.py
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

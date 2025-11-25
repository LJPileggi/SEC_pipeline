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

try:
    from src.distributed_clap_embeddings import run_distributed_slurm, run_local_multiprocess
except ImportError as e:
    # Fallback per l'esecuzione diretta o debug, ma avvisa
    print(f"Warning: Attempting local import of models. Error: {e}")
    # Se fallisce l'import da src, prova l'import diretto (se i files sono nella stessa dir)
    try:
        from distributed_clap_embeddings import run_distributed_slurm, run_local_multiprocess
    except ImportError as e_local:
        raise ImportError(f"Impossibile importare 'models' o dipendenze essenziali: {e_local}")

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
        "patience": 10, "epochs": 10, "batch_size": 2,
        "sampling_rate": 52100, "ref": 2.e-05, "noise_perc": 0.3, "seed": 1, 
        "center_freqs": [100.0, 500.0],
        "valid_cut_secs": [1]
    }
    # L'output di get_config_from_yaml in utils è una tupla:
    return (configs['classes'], configs['patience'], configs['epochs'], configs['batch_size'],
            configs['sampling_rate'], configs['ref'], configs['noise_perc'], configs['seed'],
            np.array(configs['center_freqs']), configs['valid_cut_secs']
            )

# Mock per le funzioni distribuite (previene l'uso di risorse reali)
def mock_setup_distributed_environment(rank, world_size, slurm=True):
    # Simula il device, essenziale per la logica successiva
    return torch.device('cuda:0' if slurm else 'cpu') 

def mock_join_logs_all_incomplete(*args, **kwargs):
    """
    Simula utils.join_logs restituendo un dizionario dove tutti i task sono incompleti (False).
    """
    status = {}
    classes_list = TEST_CLASSES # ['Music', 'Voices', 'Birds']
    cut_secs_list = [1.0, 3.0]
    for cut in cut_secs_list:
        for cls in classes_list:
            status[(cut, cls)] = False # False = Incompleto
    return status

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
        
        BASEDIR_RAW_TEST = os.path.join(cls.test_root_dir, 'testing', 'RAW_DATASET')
        BASEDIR_PREPROCESSED_TEST = os.path.join(cls.test_root_dir, 'testing', 'PREPROCESSED_DATASET')

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
        
        # Funzioni da utils.py
        mock_write_log = MagicMock()
        mock_join_logs = mock_join_logs_all_incomplete # Simuliamo tutti i task incompleti

        # Definisci il side effect per simulare la creazione dei file e la scrittura del log
        def mock_worker_process_side_effect(*args, **kwargs):
            tasks = args[5] # my_tasks
            rank = args[3] # rank
            for cut_secs, class_name in tasks:
                # Simula la creazione del file HDF5 di output
                h5_path = os.path.join(self.preprocessed_dir, TEST_AUDIO_FORMAT, f'{TEST_N_OCTAVE}_octave', f'cut_{cut_secs}', f'{class_name}_emb.h5')
                os.makedirs(os.path.dirname(h5_path), exist_ok=True)
                with open(h5_path, 'w') as f:
                    f.write("mock content")

                # Simula la chiamata a utils.write_log
                mock_write_log(rank=rank, 
                               cut_secs=cut_secs, 
                               class_to_process=class_name, 
                               audio_format=TEST_AUDIO_FORMAT, 
                               n_octave=TEST_N_OCTAVE)
    
        mock_worker_process.side_effect = mock_worker_process_side_effect

        # Simula il processo mp.Process in modo che chiami il worker mockato
        mock_process = MagicMock()

        # AGGIUNGIAMO TUTTI i MOCK IN UN UNICO CONTESTO 'WITH PATCH'
        with patch('src.distributed_clap_embeddings.CLAP_initializer', MagicMock()) as mock_clap_init, \
             patch('src.distributed_clap_embeddings.local_worker_process', mock_worker_process), \
             patch('src.distributed_clap_embeddings.combine_hdf5_files', mock_combine_files), \
             patch('src.distributed_clap_embeddings.mp.Manager', return_value=mock_mp_manager), \
             patch.object(mock_mp_manager, 'Queue', return_value=mock_mp_queue), \
             patch('src.distributed_clap_embeddings.mp.Process', return_value=mock_process), \
             patch('src.distributed_clap_embeddings.join_logs', mock_join_logs), \
             patch('src.distributed_clap_embeddings.write_log', mock_write_log), \
             patch('src.distributed_clap_embeddings.logging.info', MagicMock()), \
             patch('src.distributed_clap_embeddings.logging.error', MagicMock()), \
             patch('src.distributed_clap_embeddings.basedir_preprocessed', BASEDIR_PREPROCESSED_TEST), \
             patch('src.distributed_clap_embeddings.get_config_from_yaml', mock_get_config_from_yaml_data):
        
            # ESECUZIONE DEL TEST
            run_local_multiprocess(
                config_file=TEST_CONFIG_FILENAME,
                # raw_dir=self.raw_dir,
                audio_format=TEST_AUDIO_FORMAT,
                n_octave=TEST_N_OCTAVE,
                world_size=4
            )
        
            # ASSERTIONS
            self.assertEqual(mock_clap_init.call_count, 1, "CLAP_initializer deve essere chiamato una sola volta nel processo principale.")
            self.assertEqual(mock_process.call_count, 4, "Devono essere avviati 4 processi worker (world_size=4).")
            self.assertEqual(mock_worker_process.call_count, 4, "Il worker mockato deve essere chiamato una volta per ogni processo.")
            self.assertEqual(mock_write_log.call_count, 6, "write_log deve essere chiamato 6 volte (una per ogni task completato).")
            self.assertEqual(mock_combine_files.call_count, 1, "La combinazione deve avvenire una sola volta.")
        
            # Verifica l'esistenza di tutti i 6 file di output
            total_tasks = [(1.0, c) for c in TEST_CLASSES] + [(3.0, c) for c in TEST_CLASSES]
            for cut_secs, class_name in total_tasks:
                h5_path = os.path.join(self.preprocessed_dir, TEST_AUDIO_FORMAT, f'{TEST_N_OCTAVE}_octave', f'cut_{cut_secs}', f'{class_name}_emb.h5')
                self.assertTrue(os.path.exists(h5_path), f"File HDF5 mancante per task: {class_name}, cut={cut_secs}")

    def test_run_distributed_slurm_gpu_mode_rank0(self):
        """Testa il loop principale in modalità SLURM (GPU distribuita) sul Rank 0."""
        
        # DEFINIZIONE E SETUP DEI MOCK
        mock_process_class = MagicMock()
        mock_combine_files = MagicMock()
        
        # Funzioni da utils.py
        # setup_distributed_environment ritorna: device='cuda', rank=0, world_size=2
        mock_setup_dist = MagicMock(return_value=('cuda', 0, 2)) 
        mock_cleanup_dist = MagicMock()                          
        mock_write_log = MagicMock()                             
        mock_join_logs = mock_join_logs_all_incomplete           # Simuliamo tutti i task incompleti

        # Definisci il side effect per simulare la creazione del file HDF5 e la scrittura del log
        def mock_process_class_side_effect(*args, **kwargs):
            # args: (clap_model, audio_embedding, class_to_process, cut_secs, n_octave, config)
            class_name = args[2] 
            cut_secs = args[3] 
            
            # Simula la creazione del file di output
            h5_path = os.path.join(self.preprocessed_dir, TEST_AUDIO_FORMAT, f'{TEST_N_OCTAVE}_octave', f'cut_{cut_secs}', f'{class_name}_emb.h5')
            os.makedirs(os.path.dirname(h5_path), exist_ok=True)
            with open(h5_path, 'w') as f:
                f.write("mock content")

            # Simula la chiamata a utils.write_log
            mock_write_log(rank=0, 
                           cut_secs=cut_secs, 
                           class_to_process=class_name, 
                           audio_format=TEST_AUDIO_FORMAT, 
                           n_octave=TEST_N_OCTAVE) 

        mock_process_class.side_effect = mock_process_class_side_effect
        
        # AGGIUNGIAMO TUTTI I MOCK IN UN UNICO CONTESTO 'WITH PATCH'
        with patch('src.distributed_clap_embeddings.CLAP_initializer', MagicMock()) as mock_clap_init, \
             patch('src.distributed_clap_embeddings.process_class_with_cut_secs', mock_process_class), \
             patch('src.distributed_clap_embeddings.combine_hdf5_files', mock_combine_files), \
             patch('src.distributed_clap_embeddings.setup_distributed_environment', mock_setup_dist), \
             patch('src.distributed_clap_embeddings.cleanup_distributed_environment', mock_cleanup_dist), \
             patch('src.distributed_clap_embeddings.join_logs', mock_join_logs), \
             patch('src.distributed_clap_embeddings.write_log', mock_write_log), \
             patch('src.distributed_clap_embeddings.logging.info', MagicMock()), \
             patch('src.distributed_clap_embeddings.logging.error', MagicMock()), \
             patch('src.distributed_clap_embeddings.basedir_preprocessed', BASEDIR_PREPROCESSED_TEST), \
             patch('src.distributed_clap_embeddings.get_config_from_yaml', mock_get_config_from_yaml_data):
            
            # ESECUZIONE DEL TEST
            run_distributed_slurm(
                config_file=TEST_CONFIG_FILENAME,
                # raw_dir=self.raw_dir,
                audio_format=TEST_AUDIO_FORMAT,
                n_octave=TEST_N_OCTAVE
            )
            
            # ASSERTIONS
            self.assertEqual(mock_clap_init.call_count, 1, "CLAP_initializer deve essere chiamato una sola volta.")
            self.assertEqual(mock_process_class.call_count, 3, "Il Rank 0 deve eseguire 3 task.")
            self.assertEqual(mock_combine_files.call_count, 2, "La combinazione deve avvenire per i 2 cut_secs (1.0 e 3.0).")
            self.assertEqual(mock_write_log.call_count, 3, "write_log deve essere chiamato 3 volte.")
            mock_cleanup_dist.assert_called_once()
            
            # Verifica l'esistenza dei file di output per i task del rank 0
            rank0_tasks = [(1.0, 'Music'), (1.0, 'Birds'), (3.0, 'Voices')]
            for cut_secs, class_name in rank0_tasks:
                h5_path = os.path.join(self.preprocessed_dir, TEST_AUDIO_FORMAT,f'{TEST_N_OCTAVE}_octave',
                                                                f'cut_{cut_secs}', f'{class_name}_emb.h5')
                self.assertTrue(os.path.exists(h5_path), f"File HDF5 mancante per task: {class_name}, cut={cut_secs} (Rank 0)")


if __name__ == '__main__':
    # Esempio di esecuzione: python test_distributed_embeddings.py
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

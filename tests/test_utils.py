import unittest
import os
import sys
import shutil
import tempfile
import json
import glob
from unittest.mock import patch, MagicMock

# Importa i moduli necessari
import h5py
import numpy as np
import pandas as pd
import yaml
import torch
import torch.distributed as dist

# --- ASSUNZIONI DI CONSTANTI E IMPORTAZIONI ---
# Queste costanti vengono utilizzate nel tuo setup originale
TEST_CONFIG_FILENAME = 'config0.yaml'
TEST_H5_FILENAME = 'mock_audio_data.h5'
NUM_SAMPLES_H5 = 10
TEST_LOG_DIR = 'temp_logs'

# Assumendo che utils.py sia in una cartella 'src' (adatta il tuo path)
sys.path.append('.')
try:
    from src.utils import get_config_from_yaml, HDF5DatasetManager
except ImportError:
    # Fallback per l'esecuzione diretta
    print("Warning: Tentativo di importazione locale di utils.")
    # In un ambiente di test reale, queste classi dovrebbero essere importabili.
    # Definisco qui una shell delle classi e funzioni se non importabili per far girare i mock.
    class HDF5DatasetManager:
        """Shell Class for Mocking"""
        def __init__(self, *args, **kwargs):
            self.hf = None
            self.metadata_df = pd.DataFrame()
            self.audio_dset_name = 'audio_wav' # Default per il mock
            self.metadata_dset_name = 'metadata_wav' # Default per il mock
            
            # Tenta di aprire il file h5 e caricare i metadati per i test
            file_path = args[0]
            audio_format = kwargs.get('audio_format', 'wav')
            
            try:
                self.hf = h5py.File(file_path, 'r')
                self.metadata_df = pd.DataFrame(self.hf[self.metadata_dset_name][()])
                self.metadata_df['hdf5_index'] = self.metadata_df.index
                self.metadata_df.index.name = 'hdf5_index'
            except Exception as e:
                # Replica la logica di errore del Manager originale
                raise Exception(f"Errore nel Manager: {e}") from e

        def __getitem__(self, index):
            # Implementazione minimale per il test
            if self.hf and self.audio_dset_name in self.hf:
                 return self.hf[self.audio_dset_name][index]
            raise IndexError("Indice non valido o dataset audio non trovato.")
            
        def get_audio_metadata(self, hdf5_index):
            return self.metadata_df.loc[[hdf5_index]]

        def get_reproducible_permutation(self, seed: int) -> pd.DataFrame:
            # Implementazione corretta del metodo
            to_permute = self.metadata_df.copy()
            # Uso del seed direttamente come richiesto
            return to_permute.sample(frac=1, random_state=seed).reset_index(drop=False)

        def close(self):
            if self.hf and self.hf.id.valid:
                self.hf.close()
            print("HDF5 Dataset Manager chiuso.")
    
    def get_config_from_yaml(*args, **kwargs):
        """Shell Function for Mocking"""
        # Utilizza la configurazione fittizia globale per il mock
        content = TestUtils.TEST_YAML_CONTENT
        return (content['classes'], content['patience'], content['epochs'], 
                content['batch_size'], content['sampling_rate'], content['ref'], 
                content['noise_perc'], content['seed'], np.array(content['center_freqs']), 
                content['valid_cut_secs'], content['test_cut_secs'])

# ==============================================================================
# 1. FUNZIONI DI SETUP PER FILE MOCK
# ==============================================================================

def create_mock_yaml(temp_dir, config_content, filename=TEST_CONFIG_FILENAME):
    """Crea un file YAML di configurazione fittizio."""
    config_path = os.path.join(temp_dir, 'configs', filename)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.safe_dump(config_content, f)
    return config_path

def create_mock_hdf5_file(file_path, num_samples, audio_format='wav', seed=42):
    """
    Crea un file HDF5 mock con la struttura richiesta da HDF5DatasetManager.
    Restituisce i dati mock per la verifica.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    METADATA_DTYPE = np.dtype([
        ('subclass', h5py.string_dtype(encoding='utf-8')),
        ('track_name', h5py.string_dtype(encoding='utf-8')),
        ('original_index', np.int32)
    ])
    
    vlen_dtype = h5py.vlen_dtype(np.dtype('float32'))
    np.random.seed(seed)
    
    metadata_list = []
    mock_audio_list = []
    MAX_AUDIO_LEN = 52100 * 5 
    class_name = 'MockClass'

    for i in range(num_samples):
        # Dati audio VLEN
        length = np.random.randint(52100, MAX_AUDIO_LEN)
        mock_audio_data = np.random.rand(length).astype('float32') * 2 - 1
        mock_audio_list.append(mock_audio_data)
        
        # Dati metadati
        metadata_list.append((f'Subclass{i}'.encode('utf-8'), 
                              f'track_{i:03d}_{class_name}.{audio_format}'.encode('utf-8'), 
                              i))

    metadata_array = np.array(metadata_list, dtype=METADATA_DTYPE)
    
    with h5py.File(file_path, 'w') as hf:
        # Root Attributes
        hf.attrs['audio_format'] = audio_format
        hf.attrs['class'] = class_name
        hf.attrs['class_idx'] = 0
        hf.attrs['sample_rate'] = 52100 
        hf.attrs['description'] = 'Raw audio data as variable-length float32 arrays.'
        
        # Dataset Audio (VLEN)
        hf.create_dataset(f'audio_{audio_format}', data=mock_audio_list,
                                       dtype=vlen_dtype, 
                                       chunks=(1024,), maxshape=(None,))
        
        # Dataset Metadati (Strutturato)
        hf.create_dataset(f'metadata_{audio_format}', data=metadata_array)
        
    return mock_audio_list, metadata_array


# ==============================================================================
# 2. CLASSE DI TEST UNIFICATA (TestUtils)
# ==============================================================================

class TestUtils(unittest.TestCase):
    
    # Definisci il contenuto YAML come variabile di classe (come nel tuo setup)
    _yaml_content_keys = ["classes", "patience", "epochs", "batch_size", "sampling_rate",
                          "ref", "noise_perc", "seed", "center_freqs", "valid_cut_secs",
                          "test_cut_secs"]
    _yaml_content_values = [["ClassA", "ClassB"], 10, 50, 16, 52100, 2.0e-05, 0.3, 42,
                            [100.0, 500.0], [1.0, 2.0], [3.0]]
    TEST_YAML_CONTENT = {}
    for k, v in zip(_yaml_content_keys, _yaml_content_values):
        TEST_YAML_CONTENT[k] = v
    
    @classmethod
    def setUpClass(cls):
        """Setup iniziale per la configurazione YAML e HDF5."""
        
        cls.temp_dir_obj = tempfile.TemporaryDirectory()
        cls.temp_root_dir = cls.temp_dir_obj.name
        
        # Simula il contenuto del file config
        cls.config_path = create_mock_yaml(cls.temp_root_dir, cls.TEST_YAML_CONTENT, filename=TEST_CONFIG_FILENAME)
        
        # Patch per simulare che il file di config si trovi in './configs/'
        # Nota: La patch di os.path.join deve essere usata con cautela e ripristinata.
        cls.original_path_join = os.path.join
        os.path.join = lambda a, b: cls.original_path_join(cls.temp_root_dir, a, b) if (a == 'configs' or a == os.path.join(cls.temp_root_dir, 'configs')) else cls.original_path_join(a, b)

        # Creazione file HDF5 fittizio per i test dei Dataset Manager
        cls.h5_filepath_data = os.path.join(cls.temp_root_dir, 'hdf5_data', TEST_H5_FILENAME)
        # La funzione mock deve restituire i dati per la verifica nei test
        cls.mock_audio_list, cls.mock_metadata_array = create_mock_hdf5_file(
            cls.h5_filepath_data, num_samples=NUM_SAMPLES_H5, audio_format='wav'
        )
        cls.audio_format = 'wav' # Formato usato nel mock file
        
        # Inizializza l'HDF5DatasetManager e salvalo come variabile di classe
        # Patchiamo il logging per non inondare l'output durante l'inizializzazione
        with patch('logging.info'), patch('logging.error'):
             try:
                 cls.manager = HDF5DatasetManager(cls.h5_filepath_data, audio_format=cls.audio_format)
             except Exception as e:
                 cls.fail(f"Errore nell'inizializzazione di HDF5DatasetManager in setUpClass: {e}")

        # Percorso per il file HDF5 di embeddings (creato e modificato da HDF5EmbeddingDatasetsManager)
        cls.h5_filepath_embeddings = os.path.join(cls.temp_root_dir, 'hdf5_embeddings', 'embeddings.h5')


    @classmethod
    def tearDownClass(cls):
        """Pulizia finale: chiude l'HDF5 Manager e rimuove la directory temporanea."""
        # Chiude esplicitamente il file HDF5 del Manager
        if hasattr(cls, 'manager') and cls.manager.hf and cls.manager.hf.id.valid:
            cls.manager.close()
            
        cls.temp_dir_obj.cleanup()
        os.path.join = cls.original_path_join # Ripristina os.path.join
        
    def setUp(self):
        """Setup prima di ogni test."""
        self.temp_log_dir = os.path.join(self.temp_root_dir, TEST_LOG_DIR)
        shutil.rmtree(self.temp_log_dir, ignore_errors=True)
        os.makedirs(self.temp_log_dir, exist_ok=True)

        # Carica la config (necessaria per alcuni test)
        self.config = get_config_from_yaml(TEST_CONFIG_FILENAME)
        self.config_classes = self.config[0]

    # ==========================================================================
    # TEST config files e log
    # ==========================================================================

    def test_01_get_config_from_yaml(self):
        """Testa il corretto caricamento e parsing del file di configurazione."""
        config_data = get_config_from_yaml(config_file=TEST_CONFIG_FILENAME)
        self.assertIsInstance(config_data, tuple)
        for i, k in enumerate(_yaml_content_keys):
            self.assertEqual(config_data[i], TEST_YAML_CONTENT[k])


    def test_02_write_log(self):
        """
        Testa la scrittura di un singolo file di log, rispettando il formato 
        {('cut_secs', 'class') : {'process_time', 'rank'}, 'config': {kwargs}}.
        """
        log_file_base = 'test_log.json'
        
        # Simuliamo un file di log già esistente per vedere l'aggiornamento
        initial_log_data = {
            '0.5_ClassZ': {"process_time": 5.0, "rank": 0},
            'config': {"sampling_rate": 52100, "n_octave": 3}
        }
        initial_log_file = os.path.join(self.temp_log_dir, log_file_base)
        with open(initial_log_file, 'w') as f:
            json.dump(initial_log_data, f, indent=4)
        
        # Scriviamo un nuovo risultato per una task e aggiorniamo il config
        new_cut_secs_class = '1.0_ClassA'
        
        # La funzione write_log deve usare i kwargs per aggiornare la chiave 'config'
        write_log(
            log_path=os.path.join(self.temp_log_dir, log_file_base), # Nota: passo il nome file specifico per l'aggiornamento
            new_cut_secs_class=new_cut_secs_class, 
            process_time=123.45, 
            rank=0 
        )
        
        expected_log_file = os.path.join(self.temp_log_dir, log_file_base)
        self.assertTrue(os.path.exists(expected_log_file))
        
        with open(expected_log_file, 'r') as f:
            log_content = json.load(f)
            
            # 1. Verifica che i risultati della task precedente siano mantenuti
            self.assertIn('0.5_ClassZ', log_content)
            
            # 2. Verifica che i nuovi risultati della task siano stati aggiunti
            self.assertIn(new_cut_secs_class, log_content)
            self.assertEqual(log_content[new_cut_secs_class]['process_time'], 123.45)
            self.assertEqual(log_content[new_cut_secs_class]['rank'], 0)
            
            # 3. Verifica che la chiave 'config' esista e contenga i kwargs
            self.assertIn('config', log_content)
            self.assertEqual(log_content['config']['sampling_rate'], 52100)
            self.assertEqual(log_content['config']['n_octave'], 3)


    def test_03_join_logs(self):
        """Testa l'unione di più file di log nel formato richiesto."""
        
        # Scriviamo il primo log (completato)
        write_log(
            log_path=os.path.join(self.temp_log_dir, 'log_rank_0.json'),
            new_cut_secs_class='1.0_Bells',
            process_time=10.0,
            rank=0,
            sampling_rate=52100, # Questi vanno in config
            n_octave=3
        )
        
        # Scriviamo il secondo log (completato)
        write_log(
            log_path=os.path.join(self.temp_log_dir, 'log_rank_1.json'),
            new_cut_secs_class='2.0_Birds',
            process_time=20.0,
            rank=1,
            sampling_rate=52100, # Questi vanno in config
            n_octave=3 
        )

        # Esegui l'unione
        join_logs(log_dir=self.temp_log_dir)
        
        consolidated_log_path = os.path.join(self.temp_log_dir, 'log.json')
        self.assertTrue(os.path.exists(consolidated_log_path))
        
        with open(consolidated_log_path, 'r') as f:
            consolidated_log = json.load(f)

            # Verifica che le task completate siano presenti
            self.assertIn('1.0_Bells', consolidated_log)
            self.assertIn('2.0_Birds', consolidated_log)

            # Verifica la struttura delle task
            self.assertEqual(consolidated_log['1.0_Bells']['process_time'], 10.0)
            self.assertEqual(consolidated_log['2.0_Birds']['rank'], 1)

            # Verifica che la chiave 'config' sia presente e unita
            self.assertIn('config', consolidated_log)
            # Verifica che i parametri di config di entrambi i log completati siano uniti/presenti
            self.assertEqual(consolidated_log['config']['sampling_rate'], 52100)
            self.assertEqual(consolidated_log['config']['n_octave'], 3)

    # ==========================================================================
    # TEST HDF5DatasetManager (Nuove Aggiunte)
    # ==========================================================================

    def test_04_hdf5_manager_init_success(self):
        """Verifica l'inizializzazione del manager e il caricamento dei metadati."""
        # manager è stato inizializzato in setUpClass
        self.assertIsNotNone(self.manager.hf)
        self.assertTrue(self.manager.hf.id.valid)
        self.assertIsNotNone(self.manager.metadata_df)
        self.assertEqual(len(self.manager.metadata_df), NUM_SAMPLES_H5)
        self.assertTrue('hdf5_index' in self.manager.metadata_df.columns)

    def test_05_hdf5_manager_getitem_audio_access(self):
        """Verifica l'accesso rapido ai dati audio VLEN tramite l'indice HDF5 (__getitem__)."""
        # Testa un indice casuale, ad esempio 5
        test_index = 5
        audio_data = self.manager[test_index]
        
        self.assertTrue(isinstance(audio_data, np.ndarray))
        # Verifica la correttezza rispetto ai dati mock salvati
        self.assertTrue(np.allclose(audio_data, self.mock_audio_list[test_index]))
        self.assertTrue(audio_data.ndim == 1)

    def test_06_hdf5_manager_get_audio_metadata(self):
        """Verifica che get_audio_metadata restituisca il corretto sub-DataFrame per un indice HDF5."""
        
        hdf5_index = 2
        metadata_df = self.manager.get_audio_metadata(hdf5_index)
        
        self.assertTrue(isinstance(metadata_df, pd.DataFrame))
        self.assertEqual(len(metadata_df), 1)
        self.assertEqual(metadata_df.iloc[0]['hdf5_index'], hdf5_index)
        
        # Verifica il track_name
        expected_track_name = self.mock_metadata_array[hdf5_index]['track_name'].decode('utf-8')
        self.assertEqual(metadata_df.iloc[0]['track_name'].decode('utf-8'), expected_track_name)
        
    def test_07_hdf5_manager_get_reproducible_permutation(self):
        """Verifica che la permutazione sia riproducibile e usi il 'seed' correttamente."""
        
        fixed_seed = 101 # Usiamo un seed diverso da quello di config per sicurezza
        
        # Esegui due volte con lo stesso seed
        permuted_df_1 = self.manager.get_reproducible_permutation(fixed_seed)
        permuted_df_2 = self.manager.get_reproducible_permutation(fixed_seed)
        
        # Verifica la dimensione e l'integrità dei dati
        self.assertEqual(len(permuted_df_1), NUM_SAMPLES_H5)
        self.assertEqual(set(permuted_df_1['hdf5_index']), set(range(NUM_SAMPLES_H5)))
        
        # Verifica la riproducibilità (l'ordine delle righe deve essere identico)
        pd.testing.assert_frame_equal(permuted_df_1.reset_index(drop=True), permuted_df_2.reset_index(drop=True))
        
        # Esegui con un seed diverso per assicurare che ci sia una permutazione
        permuted_df_3 = self.manager.get_reproducible_permutation(fixed_seed + 1)
        
        # Verifica che l'ordine sia diverso
        with self.assertRaises(AssertionError):
            pd.testing.assert_frame_equal(permuted_df_1.reset_index(drop=True), permuted_df_3.reset_index(drop=True))

    def test_08_HDF5DatasetManager_close(self):
        """Testa la chiusura del file HDF5 handle."""
        self.assertFalse(self.manager.h5_file_handle.closed)
        manager.close()
        self.assertTrue(self.manager.h5_file_handle.closed)

    def test_09_HDF5DatasetManager_del(self):
        """Testa che il file HDF5 venga chiuso quando l'oggetto è distrutto."""
        h5_handle = self.manager.h5_file_handle
        self.assertFalse(h5_handle.closed)
        
        # Elimina il riferimento al manager per forzare la chiamata a __del__
        del self.manager
        # Python potrebbe non chiamare __del__ immediatamente, forziamo il GC o
        # attendiamo un momento. Per test unitari, il modo più robusto è controllare
        # il mock se avessimo patchato l'handle. Senza patch, possiamo solo sperare
        # o usare gc.collect() se strettamente necessario (ma sconsigliato nei test).
        
        # Per un test più deterministico, patchiamo h5_file_handle.close
        with patch.object(h5_handle, 'close') as mock_close:
            manager_del_test = HDF5DatasetManager(h5_file_path=self.h5_filepath_data, audio_format='wav')
            manager_del_test.h5_file_handle = h5_handle # Sostituiamo l'handle con quello mockato
            del manager_del_test
            # Se mock_close non viene chiamato, il test fallirà.
            # Questo verifica che __del__ tenti di chiudere l'handle.
            mock_close.assert_called_once()

    # ==========================================================================
    # Test Classe HDF5EmbeddingDatasetsManager
    # ==========================================================================
    
    def test_10_HDF5EmbeddingDatasetsManager_init(self):
        """Testa l'inizializzazione di HDF5EmbeddingDatasetsManager."""
        # Non creiamo il file embeddings.h5 in setUpClass per questo test, lo faremo dopo
        # o lo mockiamo se stiamo testando la modalità 'r'.
        # Per il primo test init, creiamolo vuoto per la modalità 'w' (se fosse supportata)
        
        # La versione fornita non ha un 'w' mode diretto, ma usa 'a' implicitamente per initialize_hdf5.
        # Per testare l'init, simuliamo un file HDF5 già esistente con alcuni gruppi.
        # h5_path punta al file dove saranno salvati gli embeddings.
        
        # Creiamo un file HDF5 vuoto per initialize_hdf5
        if os.path.exists(self.h5_filepath_embeddings):
            os.remove(self.h5_filepath_embeddings)
        
        manager = HDF5EmbeddingDatasetsManager(h5_path=self.h5_filepath_embeddings, mode='a') # 'a' è implicito con initialize_hdf5
        self.assertEqual(manager.h5_path, self.h5_filepath_embeddings)
        self.assertEqual(manager.mode, 'a')
        self.assertEqual(manager.partitions, {'classes', 'splits'})
        self.assertIsNotNone(manager.h5_file_handle)
        
        manager.close() # Chiudi l'handle per evitare problemi


    def test_11_HDF5EmbeddingDatasetsManager_len(self):
        """Testa la lunghezza del dataset (numero totale di embeddings)."""
        # Creiamo un file HDF5 con dati reali per il manager embeddings
        if os.path.exists(self.h5_filepath_embeddings):
            os.remove(self.h5_filepath_embeddings)
            
        manager = HDF5EmbeddingDatasetsManager(h5_path=self.h5_filepath_embeddings, mode='a')
        # Inizializza il file con alcuni dati
        embedding_dim = TEST_EMBED_DIM
        spec_shape = (128, 1024)
        audio_format = 'wav'
        cut_secs = 1.0
        n_octave = 3
        sample_rate = 52100
        seed = 42
        noise_perc = 0.3
        split = 'train'
        class_name = 'ClassA'
        
        manager.initialize_hdf5(embedding_dim, spec_shape, audio_format, cut_secs, n_octave, 
                                 sample_rate, seed, noise_perc, split, class_name)
        
        # Aggiungiamo dati fittizi
        for i in range(5):
            embedding = np.random.rand(embedding_dim).astype('float32')
            spectrogram = np.random.rand(*spec_shape).astype('float32')
            hash_keys = f'hash_{i}'
            track_name = f'track_{i}'
            manager.add_to_data_buffer(embedding, spectrogram, hash_keys, track_name, class_=class_name)
        manager.flush_buffers() # Scrive i 5 elementi
        
        self.assertEqual(len(manager), 5) # Dovrebbe contare i 5 elementi appena aggiunti
        manager.close() # Chiudi


    def test_12_HDF5EmbeddingDatasetsManager_getitem(self):
        """Testa il recupero di un singolo elemento (embedding e metadati) dal manager."""
        if os.path.exists(self.h5_filepath_embeddings):
            os.remove(self.h5_filepath_embeddings)
            
        manager = HDF5EmbeddingDatasetsManager(h5_path=self.h5_filepath_embeddings, mode='a')
        manager.initialize_hdf5(TEST_EMBED_DIM, (128,1024), 'wav', 1.0, 3, 52100, 42, 0.3, 'train', 'ClassA')
        
        expected_embedding = np.random.rand(TEST_EMBED_DIM).astype('float32')
        expected_spectrogram = np.random.rand(128,1024).astype('float32')
        expected_track_name = 'test_track_0'
        
        manager.add_to_data_buffer(expected_embedding, expected_spectrogram, 'hash_0', expected_track_name, class_='ClassA')
        manager.flush_buffers()
        
        embedding, metadata = manager[0]
        
        self.assertIsInstance(embedding, torch.Tensor)
        self.assertEqual(embedding.shape, (TEST_EMBED_DIM,))
        self.assertTrue(np.array_equal(embedding.numpy(), expected_embedding)) # Confronta i numpy array sottostanti
        
        self.assertIsInstance(metadata, pd.Series)
        self.assertEqual(metadata['track_names'], expected_track_name)
        self.assertEqual(metadata['classes'], 'ClassA')
        
        manager.close()


    def test_13_HDF5EmbeddingDatasetsManager_set_dataset_format(self):
        """Testa l'impostazione del formato dei dataset HDF5."""
        if os.path.exists(self.h5_filepath_embeddings):
            os.remove(self.h5_filepath_embeddings)
        
        manager = HDF5EmbeddingDatasetsManager(h5_path=self.h5_filepath_embeddings, mode='a')
        manager._set_dataset_format(embedding_dim=TEST_EMBED_DIM, spec_shape=(128, 1024))
        
        # Verifica che gli attributi siano stati impostati correttamente
        self.assertEqual(manager.embedding_dim, TEST_EMBED_DIM)
        self.assertEqual(manager.spec_shape, (128, 1024))
        self.assertEqual(manager.data_dtypes['embeddings'], 'f4')
        self.assertEqual(manager.data_dtypes['spectrograms'], 'f4')
        
        manager.close()


    def test_14_HDF5EmbeddingDatasetsManager_initialize_hdf5(self):
        """Testa l'inizializzazione della struttura HDF5."""
        if os.path.exists(self.h5_filepath_embeddings):
            os.remove(self.h5_filepath_embeddings)
            
        manager = HDF5EmbeddingDatasetsManager(h5_path=self.h5_filepath_embeddings, mode='a')
        
        # Chiamata alla funzione
        manager.initialize_hdf5(
            embedding_dim=TEST_EMBED_DIM, 
            spec_shape=(128, 1024), 
            audio_format='wav', 
            cut_secs=1.0, 
            n_octave=3, 
            sample_rate=52100, 
            seed=42, 
            noise_perc=0.3, 
            split='train', 
            class_name='ClassA'
        )
        
        # Verifica che il gruppo e i dataset siano stati creati
        with h5py.File(self.h5_filepath_embeddings, 'r') as f:
            self.assertIn(group_path, f)
            self.assertIn('embeddings', f['embedding_dataset'])
            self.assertIn('spectrograms', f['embedding_dataset'])
            self.assertIn('ID', f['embedding_dataset'])
            self.assertIn('track_names', f['embedding_dataset'])
            
            # Verifica attributi
            self.assertEqual(f[group_path].attrs['n_octave'], 3)
            
        manager.close()


    def test_15_HDF5EmbeddingDatasetsManager_add_to_data_buffer(self):
        """Testa l'aggiunta di dati al buffer temporaneo."""
        if os.path.exists(self.h5_filepath_embeddings):
            os.remove(self.h5_filepath_embeddings)
            
        manager = HDF5EmbeddingDatasetsManager(h5_path=self.h5_filepath_embeddings, mode='a')
        manager.initialize_hdf5(TEST_EMBED_DIM, (128,1024), 'wav', 1.0, 3, 52100, 42, 0.3, 'train', 'ClassA')
        
        embedding = np.random.rand(TEST_EMBED_DIM).astype('f4')
        spectrogram = np.random.rand(128, 1024).astype('f4')
        hash_keys = 'test_hash'
        track_name = 'test_track'
        
        manager.add_to_data_buffer(embedding, spectrogram, hash_keys, track_name, class_='ClassA')
        
        self.assertEqual(len(manager.embeddings_buffer), 1)
        self.assertTrue(np.array_equal(manager.embeddings_buffer[0], embedding))
        
        manager.close()


    def test_16_HDF5EmbeddingDatasetsManager_flush_buffers(self):
        """Testa lo svuotamento dei buffer e la scrittura su HDF5."""
        if os.path.exists(self.h5_filepath_embeddings):
            os.remove(self.h5_filepath_embeddings)
            
        manager = HDF5EmbeddingDatasetsManager(h5_path=self.h5_filepath_embeddings, mode='a')
        manager.initialize_hdf5(TEST_EMBED_DIM, (128,1024), 'wav', 1.0, 3, 52100, 42, 0.3, 'train', 'ClassA')
        
        # Aggiungi alcuni dati al buffer
        for i in range(3):
            manager.add_to_data_buffer(
                np.random.rand(TEST_EMBED_DIM).astype('f4'),
                np.random.rand(128, 1024).astype('f4'),
                f'hash_{i}', f'track_{i}', class_='ClassA'
            )
        
        self.assertEqual(len(manager.embeddings_buffer), 3)
        
        manager.flush_buffers()
        
        # I buffer dovrebbero essere vuoti dopo il flush
        self.assertEqual(len(manager.embeddings_buffer), 0)
        
        # Verifica che i dati siano stati scritti nel file HDF5
        with h5py.File(self.h5_filepath_embeddings, 'r') as f:
            self.assertEqual(f['embedding_dataset']['embeddings'].shape, (3, TEST_EMBED_DIM))
            self.assertEqual(f['embedding_dataset']['track_names'].shape, (3,))
            
        manager.close()


    def test_17_HDF5EmbeddingDatasetsManager_extend_dataset(self):
        """Testa l'estensione di un dataset esistente."""
        if os.path.exists(self.h5_filepath_embeddings):
            os.remove(self.h5_filepath_embeddings)
            
        manager = HDF5EmbeddingDatasetsManager(h5_path=self.h5_filepath_embeddings, mode='a')
        manager.initialize_hdf5(TEST_EMBED_DIM, (128,1024), 'wav', 1.0, 3, 52100, 42, 0.3, 'train', 'ClassA')
        
        # Aggiungi 2 dati iniziali e scrivi
        for i in range(2):
            manager.add_to_data_buffer(
                np.random.rand(TEST_EMBED_DIM).astype('f4'), np.random.rand(128, 1024).astype('f4'),
                f'hash_initial_{i}', f'track_initial_{i}', class_='ClassA'
            )
        manager.flush_buffers()
        
        # Crea nuovi dati da estendere
        new_embeddings = np.array([np.random.rand(TEST_EMBED_DIM) for _ in range(2)], dtype='f4')
        new_spectrograms = np.array([np.random.rand(128, 1024) for _ in range(2)], dtype='f4')
        new_hash_keys = np.array(['hash_new_0', 'hash_new_1'], dtype=h5py.string_dtype(encoding='utf-8'))
        new_track_names = np.array(['track_new_0', 'track_new_1'], dtype=h5py.string_dtype(encoding='utf-8'))
        new_class_names = np.array(['classA', 'classB'], dtype=h5py.string_dtype(encoding='utf-8'))
        new_subclass_names = np.array(['subclassA', 'subclassB'], dtype=h5py.string_dtype(encoding='utf-8'))
        
        new_data = {
            'embeddings': new_embeddings,
            'spectrograms': new_spectrograms,
            'ID': new_hash_keys,
            'track_names': new_track_names,
            'classes': new_class_names,
            'subclasses': new_subclass_names
        }
        
        # Estendi il dataset
        manager.extend_dataset(new_data)
        
        # Verifica la nuova dimensione
        with h5py.File(self.h5_filepath_embeddings, 'r') as f:
            self.assertEqual(f['embedding_dataset']['embeddings'].shape, (4, TEST_EMBED_DIM)) # 2 iniziali + 2 nuovi
            self.assertEqual(f['embedding_dataset']['track_name'].shape, (4,))
            
        manager.close()


    def test_18_HDF5EmbeddingDatasetsManager_close(self):
        """Testa la chiusura del file HDF5 handle del manager."""
        if os.path.exists(self.h5_filepath_embeddings):
            os.remove(self.h5_filepath_embeddings)
        manager = HDF5EmbeddingDatasetsManager(h5_path=self.h5_filepath_embeddings, mode='a')
        self.assertFalse(manager.h5_file_handle.closed)
        manager.close()
        self.assertTrue(manager.h5_file_handle.closed)


    def test_19_HDF5EmbeddingDatasetsManager_del(self):
        """Testa che il file HDF5 sia chiuso quando l'oggetto manager viene distrutto."""
        if os.path.exists(self.h5_filepath_embeddings):
            os.remove(self.h5_filepath_embeddings)
        manager = HDF5EmbeddingDatasetsManager(h5_path=self.h5_filepath_embeddings, mode='a')
        h5_handle = manager.h5_file_handle
        self.assertFalse(h5_handle.closed)
        
        with patch.object(h5_handle, 'close') as mock_close:
            manager_del_test = HDF5EmbeddingDatasetsManager(h5_path=self.h5_filepath_embeddings, mode='a')
            manager_del_test.h5_file_handle = h5_handle # Associa il mock handle
            del manager_del_test
            mock_close.assert_called_once()

    # ==========================================================================
    # Test Funzione combine_hdf5_files
    # ==========================================================================

    @patch('src.utils.HDF5EmbeddingDatasetsManager')
    @patch('src.utils.h5py.File')
    @patch('src.utils.glob.glob')
    def test_20_combine_hdf5_files(self, mock_glob, mock_h5py_file, mock_HDF5EmbeddingDatasetsManager):
        """Testa la funzione che combina file HDF5 in un unico file di embeddings,
           usando un dataset strutturato per i file di input e un cut_secs omogeneo."""
        
        # -------------------------------------------------------------------------
        # VARIABILI DI CONFIGURAZIONE FISSE PER IL TEST
        # -------------------------------------------------------------------------
        
        # Valori di configurazione fissi (potrebbero essere definiti come attributi di classe)
        # Assumi che TEST_EMBED_DIM sia definito come attributo della classe
        TEST_EMBED_DIM = 1024 
        SPEC_SHAPE = (128, 1024)
        NUM_RECORDS = 5
        
        # CUT_SECS FISSO: Tutti i file di input e l'output devono usare questo valore
        FIXED_CUT_SECS = 1.0 
        CUT_SECS_LIST = [FIXED_CUT_SECS] # La lista passata alla funzione deve contenere solo cut_secs omogenei.

        # La lista di file che glob deve trovare (rimosso ClassA_2.0 per coerenza di cut_secs)
        MOCK_FILE_PATHS = [
            os.path.join(self.temp_root_dir, 'ClassA_1.0_wav.h5'),
            os.path.join(self.temp_root_dir, 'ClassB_1.0_wav.h5'),
        ]
        
        # Definisce la struttura (dtype) del dataset HDF5 di INPUT 
        input_dtype = np.dtype([
            ('ID', 'S100'), 
            ('embeddings', (np.float64, (TEST_EMBED_DIM,))),
            ('spectrograms', (np.float64, SPEC_SHAPE)),
            ('track_names', 'S100'),
            ('subclasses', 'S100'), 
        ])
        
        # -------------------------------------------------------------------------
        # 1. Configura il mock per glob.glob
        # -------------------------------------------------------------------------
        mock_glob.return_value = MOCK_FILE_PATHS
        
        # -------------------------------------------------------------------------
        # 2. Crea file HDF5 fittizi con un dataset strutturato (CUT_SECS OMOGENEO)
        # -------------------------------------------------------------------------
        for f_path in MOCK_FILE_PATHS:
            # Estrai le informazioni dal nome del file per gli attributi HDF5
            filename = os.path.basename(f_path)
            class_name = filename.split('_')[0]
            
            # 1. Prepara i dati
            embeddings_data = np.random.rand(NUM_RECORDS, TEST_EMBED_DIM)
            spectrograms_data = np.random.rand(NUM_RECORDS, SPEC_SHAPE[0], SPEC_SHAPE[1])
            hash_keys = np.array([f'h{j+1}_{class_name}'.encode('utf-8') for j in range(NUM_RECORDS)], dtype='S100')
            track_names = np.array([f't{j+1}_{class_name}'.encode('utf-8') for j in range(NUM_RECORDS)], dtype='S100')
            subclasses = np.array([f's{j+1}'.encode('utf-8') for j in range(NUM_RECORDS)], dtype='S100')
            
            # 2. Popola il record array
            mock_data = np.empty(NUM_RECORDS, dtype=input_dtype)
            mock_data['ID'] = hash_keys
            mock_data['embeddings'] = embeddings_data
            mock_data['spectrograms'] = spectrograms_data
            mock_data['track_names'] = track_names
            mock_data['subclasses'] = subclasses
            
            # 3. Scrivi nel file HDF5
            with h5py.File(f_path, 'w') as f:
                # Crea il dataset strutturato UNICO
                f.create_dataset('embedding_dataset', data=mock_data, compression='gzip')
                # ATTRIBUTO CRUCIALE: cut_secs è FISSO per tutti i file da combinare
                f.attrs['class'] = class_name
                f.attrs['cut_secs'] = FIXED_CUT_SECS 
                f.attrs['split'] = 'train' 
                
        # -------------------------------------------------------------------------
        # 3. Configura il mock di HDF5EmbeddingDatasetsManager per tracciare le chiamate
        # -------------------------------------------------------------------------
        mock_manager_instance = MagicMock()
        mock_HDF5EmbeddingDatasetsManager.return_value = mock_manager_instance
        
        # 4. Esegui la funzione
        combine_hdf5_files(
            root_dir=self.temp_root_dir, 
            cut_secs_list=CUT_SECS_LIST, # Passiamo solo 1.0
            audio_format='wav', 
            splits_list=['train'],
            embedding_dim=TEST_EMBED_DIM,
            spec_shape=SPEC_SHAPE
        )
        
        # -------------------------------------------------------------------------
        # 5. Verifica le chiamate al manager e al suo metodo extend_dataset
        # -------------------------------------------------------------------------
        
        # Inizializzazione (una volta per l'HDF5 di output)
        mock_manager_instance.initialize_hdf5.assert_called_once_with(
            TEST_EMBED_DIM, 
            SPEC_SHAPE, 
            'wav', 
            FIXED_CUT_SECS, # cut_secs fisso (1.0)
            3, # Assumiamo questi valori provengano dalla config (n_octave)
            52100, # (sample_rate)
            42, # (seed)
            0.3, # (noise_perc)
            'train', 
            class_name=None # Il file combinato non è specifico per una classe
        )
        
        # Due file di input (ClassA_1.0, ClassB_1.0), due chiamate a extend_dataset.
        self.assertEqual(mock_manager_instance.extend_dataset.call_count, 2) 
        
        # Verifica che il metodo close sia chiamato
        mock_manager_instance.close.assert_called_once()
        
        # Pulisci i file fittizi creati per questo test
        for f_path in MOCK_FILE_PATHS:
            if os.path.exists(f_path):
                os.remove(f_path)

    @patch('src.utils.get_track_reproducibility_parameters')
    @patch('src.utils.HDF5DatasetManager')
    @patch('src.utils.HDF5EmbeddingDatasetsManager')
    def test_reconstruct_tracks_from_embeddings_full_logic(self, MockEmbManager, MockTrackManager, mock_get_params):
        """Testa la ricostruzione completa della traccia, inclusa la logica di offset e rumore riproducibili."""
        
        # -------------------------------------------------------------------------
        # 1. SETUP DELLE COSTANTI E DEI PARAMETRI DI INPUT
        # -------------------------------------------------------------------------
        
        # Metadati del file di embedding (da MockEmbManager.hf.attrs)
        CLASS_NAME = 'ClassA'
        CLASSES_LIST = [CLASS_NAME, 'ClassB']
        AUDIO_FORMAT = 'wav'
        CUT_SECS = 2.0
        SAMPLE_RATE = 52100
        SEED = 42
        NOISE_PERC = 0.3
        WINDOW_SIZE = int(CUT_SECS * SAMPLE_RATE) # 104200

        # Parametri della traccia da ricostruire (estratti dalla stringa indice)
        CLASS_IDX = 0 
        HDF5_TRACK_INDEX = 5
        BUCKET = 1
        ROUND_ = 2      # Determina l'offset random: loop = ROUND_ + 1 = 3 volte
        RESULTS = 3     # Determina il rumore random: loop = RESULTS = 3 volte
        
        # Stringa indice fittizia
        idx_str = f'{CLASS_IDX}_{HDF5_TRACK_INDEX}_{BUCKET}_{ROUND_}_{RESULTS}'
        idx_list = [idx_str]

        # Percorsi mock
        EMB_PATH = os.path.join(self.temp_root_dir, 'embeddings_split.h5')
        RAW_DIR = os.path.join(self.temp_root_dir, f'raw_{AUDIO_FORMAT}')
        BASE_TRACKS_DIR = self.temp_root_dir # Usiamo la root temporanea
        
        # -------------------------------------------------------------------------
        # 2. CONFIGURAZIONE DEI MOCK
        # -------------------------------------------------------------------------
        
        # 2a. Mock HDF5EmbeddingDatasetsManager (metadati globali e classi)
        mock_emb_manager_instance = MagicMock()
        MockEmbManager.return_value = mock_emb_manager_instance
        
        mock_emb_manager_instance.hf.attrs = {
            'audio_format': AUDIO_FORMAT, 'cut_secs': CUT_SECS, 'sample_rate': SAMPLE_RATE, 
            'noise_perc': NOISE_PERC, 'seed': SEED
        }
        # Simula la lettura della lista di classi
        mock_class_data = MagicMock()
        mock_class_data.unique.return_value.sort.return_value = CLASSES_LIST
        # Assicura che l'accesso a hf['embedding_dataset']['classes'] ritorni l'oggetto mock
        mock_emb_manager_instance.hf.__getitem__.return_value.__getitem__.return_value = mock_class_data

        # 2b. Mock HDF5DatasetManager (tracce audio grezze)
        mock_track_manager_instance = MagicMock()
        MockTrackManager.return_value = mock_track_manager_instance
        
        # Dati audio grezzi fittizi (Lunghi a sufficienza per offset)
        TRACK_DURATION_SECS = 5.0
        TRACK_LENGTH = int(TRACK_DURATION_SECS * SAMPLE_RATE) # 260500
        np.random.seed(1234) # Seed fisso per i dati di input, non per la riproducibilità del rumore
        MOCK_ORIGINAL_TRACK = np.random.rand(TRACK_LENGTH).astype('float32') * 0.5
        
        # Simula l'accesso all'audio track: original_track = hdf5_class_tracks[repr_params['hdf5_index']]
        mock_track_manager_instance.__getitem__.return_value = MOCK_ORIGINAL_TRACK
        
        # 2c. Mock get_track_reproducibility_parameters
        mock_get_params.return_value = {
            'class_idx': CLASS_IDX, 'hdf5_index': HDF5_TRACK_INDEX, 
            'bucket': BUCKET, 'round_': ROUND_, 'results': RESULTS
        }

        # -------------------------------------------------------------------------
        # 3. CALCOLO DELL'OUTPUT ATTESO (Logica di riproducibilità simulata)
        # -------------------------------------------------------------------------

        # 3a. Calcolo del CLASS_SEED (deve essere identico alla funzione)
        class_hash = hash(CLASSES_LIST[CLASS_IDX]) % 10000000
        CLASS_SEED = SEED + class_hash
        
        # 3b. Inizializza i generatori RNG con il CLASS_SEED
        offset_rng = np.random.default_rng(CLASS_SEED)
        noise_rng = np.random.default_rng(CLASS_SEED)
        
        # 3c. Simula il calcolo dell'Offset
        offset = 0
        max_offset = MOCK_ORIGINAL_TRACK.shape[0] - WINDOW_SIZE
        if max_offset > 0:
            for _ in range(ROUND_ + 1): 
                # L'offset finale è l'ultimo valore generato dal loop
                offset = offset_rng.integers(0, max_offset)

        # 3d. Simula il taglio (Cut Track)
        start = BUCKET * WINDOW_SIZE + offset
        end = start + WINDOW_SIZE
        cut_track = MOCK_ORIGINAL_TRACK[start:end]

        # 3e. Simula il Padding (se necessario)
        if len(cut_track) < WINDOW_SIZE:
             pad_length = WINDOW_SIZE - len(cut_track)
             cut_track = np.pad(cut_track, (0, pad_length), 'constant')

        # 3f. Simula la generazione di Rumore e la Traccia finale
        abs_cut_track = np.abs(cut_track)
        max_threshold = np.mean(abs_cut_track)
        
        noise = None
        for _ in range(RESULTS): 
            # Il rumore finale è l'ultimo valore generato dal loop
            noise = noise_rng.uniform(-max_threshold, max_threshold, cut_track.shape)
            
        expected_reconstr_track = (1 - NOISE_PERC) * cut_track + NOISE_PERC * noise

        # -------------------------------------------------------------------------
        # 4. ESECUZIONE E ASSERZIONI
        # -------------------------------------------------------------------------

        # La funzione deve essere disponibile nel contesto di esecuzione (es. in src.utils)
        from src.utils import reconstruct_tracks_from_embeddings
        
        tracks_reconstructed = reconstruct_tracks_from_embeddings(BASE_TRACKS_DIR, EMB_PATH, idx_list)
        
        # 4a. Verifica delle chiamate alle funzioni helper
        mock_get_params.assert_called_with(idx_str)
        
        # 4b. Verifica che i manager siano stati usati correttamente
        # L'Embedding Manager è istanziato una volta
        MockEmbManager.assert_called_once_with(EMB_PATH, 'r', ('splits',))
        # Il Track Manager è istanziato una volta, con il percorso corretto
        MockTrackManager.assert_called_once_with(
            os.path.join(BASE_TRACKS_DIR, f'raw_{AUDIO_FORMAT}', f'{CLASS_NAME}_{AUDIO_FORMAT}_dataset.h5'), 
            AUDIO_FORMAT
        )
        # Il metodo close() deve essere chiamato sul manager dei dati audio
        mock_emb_manager_instance.close.assert_called_once()
        mock_track_manager_instance.close.assert_called_once()
        
        # 4c. Verifica dell'output
        self.assertIn(idx_str, tracks_reconstructed)
        self.assertEqual(len(tracks_reconstructed[idx_str]), WINDOW_SIZE)
        
        # Verifica del contenuto: la traccia ricostruita deve essere identica all'output atteso
        self.assertTrue(np.allclose(tracks_reconstructed[idx_str], expected_reconstr_track, atol=1e-5))


    def test_22_setup_environ_vars(self):
        """Testa l'impostazione delle variabili d'ambiente per il DDP (SLURM/locale)."""
        # Test SLURM
        with patch.dict('os.environ', {'SLURM_PROCID': '2', 'SLURM_NTASKS': '8'}, clear=True):
            rank, world_size = setup_environ_vars(slurm=True)
            self.assertEqual(rank, 2)
            self.assertEqual(world_size, 8)
        
        # Test Locale (default)
        with patch.dict('os.environ', {}, clear=True):
            rank, world_size = setup_environ_vars(slurm=False)
            self.assertEqual(os.environ.get('MASTER_ADDR'), 'localhost')
            self.assertEqual(os.environ.get('MASTER_PORT'), '29500')
            self.assertIsNone(rank) # setup_environ_vars non restituisce rank/world_size per locale
            self.assertIsNone(world_size)

    @patch('src.utils.dist.init_process_group', MagicMock())
    @patch('src.utils.torch.cuda.is_available', return_value=True)
    @patch('src.utils.torch.cuda.set_device', MagicMock())
    @patch('src.utils.torch.device', MagicMock(side_effect=lambda x: MagicMock(type='cuda',
                                index=int(x) if isinstance(x, str) and 'cuda' in x else x)))
    @patch('src.utils.logging.info', MagicMock())
    def test_23_setup_distributed_environment(self, mock_log_info, mock_torch_device,
                                mock_set_device, mock_is_available, mock_init_pg):
        """Testa l'inizializzazione dell'ambiente distribuito (DDP)."""
        
        # Test SLURM con GPU
        device = setup_distributed_environment(rank=1, world_size=4, slurm=True)
        mock_init_pg.assert_called_once_with("nccl", rank=1, world_size=4)
        mock_set_device.assert_called_once_with(1)
        self.assertEqual(device.type, 'cuda')
        self.assertEqual(device.index, 1)
        mock_init_pg.reset_mock() # Reset per il prossimo test

        # Test Locale con CPU (forzando is_available=False)
        mock_is_available.return_value = False
        device_cpu = setup_distributed_environment(rank=0, world_size=1, slurm=False)
        mock_init_pg.assert_called_once_with("gloo", rank=0, world_size=1)
        self.assertEqual(device_cpu.type, 'cpu')
        self.assertFalse(mock_set_device.called) # set_device non dovrebbe essere chiamato per CPU
        mock_is_available.return_value = True # Ripristina per altri test se necessario


    @patch('src.utils.dist.barrier', MagicMock())
    @patch('src.utils.dist.destroy_process_group', MagicMock())
    @patch('src.utils.logging.info', MagicMock())
    @patch('src.utils.os.environ.get', return_value='0') # Mock del rank globale per il logging
    def test_24_cleanup_distributed_environment(self, mock_os_get_environ, mock_log_info, mock_destroy, mock_barrier):
        """Testa la corretta pulizia dell'ambiente distribuito."""
        cleanup_distributed_environment()
        mock_barrier.assert_called_once()
        mock_destroy.assert_called_once()
        mock_log_info.assert_called()


if __name__ == '__main__':
    # Necessario per i test che usano os.path.join con la configurazione YAML
    if 'src' in os.listdir('.'):
        os.chdir(os.path.join(os.getcwd(), 'src')) 
    elif 'utils.py' not in os.listdir('.'):
         print("WARNING: Esegui i test dalla root o definisci correttamente i percorsi.")
         
    # Uso diretto di unittest.main() è più comune per i runner di test
    unittest.main()

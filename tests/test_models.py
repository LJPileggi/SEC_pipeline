import unittest
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock
sys.path.append('.')

# Importa i moduli necessari
# Assumendo che models.py sia in src
try:
    from src.models import CLAP_initializer, FinetunedModel, spectrogram_n_octaveband_generator
    # Assicurati che librosa e msclap siano importabili nel test
    import librosa
    import msclap
except ImportError as e:
    # Fallback per l'esecuzione diretta o debug, ma avvisa
    print(f"Warning: Attempting local import of models. Error: {e}")
    # Se fallisce l'import da src, prova l'import diretto (se i files sono nella stessa dir)
    try:
        from models import CLAP_initializer, FinetunedModel, spectrogram_n_octaveband_generator
        import librosa
        import msclap
    except ImportError as e_local:
        raise ImportError(f"Impossibile importare 'models' o dipendenze essenziali: {e_local}")


# ==============================================================================
# 1. FUNZIONI DI MOCKING E SETUP
# ==============================================================================

# Simula l'esistenza delle variabili d'ambiente per i percorsi dei pesi
def mock_getenv_clap_paths(key):
    """Simula os.getenv per i percorsi di CLAP."""
    if key == "LOCAL_CLAP_WEIGHTS_PATH":
        return "/tmp/CLAP_weights_2023.pth" # Un percorso fittizio ma esistente
    if key == "CLAP_TEXT_ENCODER_PATH":
        return "/tmp/roberta-base" # Un percorso fittizio ma esistente per l'encoder text
    return None

# MOCK della classe CLAP esterna (msclap.CLAP)
# Simula il suo comportamento essenziale per CLAP_initializer
class MockCLAP:
    def __init__(self, version, use_cuda, am_path, lm_path, download_if_missing=True):
        # Questi assert verificano che CLAP_initializer passi gli argomenti corretti
        self.version = version
        self.use_cuda = use_cuda
        self.am_path = am_path
        self.lm_path = lm_path
        self.model = MagicMock(spec=nn.Module) # Mock del sottocomponente 'model' per 'parameters()'
        self.model.parameters.return_value = [torch.nn.Parameter(torch.randn(10))] # Mock dei parametri

    def get_audio_embedding(self, audio_data, resample=False, use_tensor=False):
        # Simula l'output dell'embedding audio [Batch_size, Embed_Dim]
        if isinstance(audio_data, list):
             batch_size = len(audio_data)
        elif isinstance(audio_data, np.ndarray) and audio_data.ndim > 1:
             batch_size = audio_data.shape[0]
        else: # Assumiamo un singolo audio
             batch_size = 1
        return torch.randn(batch_size, 1024) # Embedding CLAP standard

# ==============================================================================
# 2. CLASSE DI TEST PRINCIPALE
# ==============================================================================

class TestModels(unittest.TestCase):
    
    # --------------------------------------------------------------------------
    # 2.1 Test CLAP_initializer
    # --------------------------------------------------------------------------

    @patch('os.path.exists', return_value=True) # I file dei pesi ESISTONO
    @patch('os.getenv', side_effect=mock_getenv_clap_paths)
    @patch('src.models.CLAP', new=MockCLAP) # Sostituisce la classe msclap.CLAP con il nostro mock
    def test_clap_initializer_success_cpu(self, mock_getenv, mock_exists):
        """Testa l'inizializzazione di CLAP su CPU con percorsi corretti (mockati)."""
        
        # Test con use_cuda=False (CPU)
        clap_model, audio_embedding_func, original_parameters = CLAP_initializer(device='cpu', use_cuda=False)
        
        # 1. Verifica che l'inizializzazione avvenga senza errori
        self.assertIsNotNone(clap_model)
        self.assertIsNotNone(audio_embedding_func)
        self.assertIsNotNone(original_parameters) 
        
        # 2. Verifica che 'clap_model' sia il nostro MockCLAP e che abbia ricevuto gli argomenti corretti
        self.assertIsInstance(clap_model, MockCLAP)
        self.assertEqual(clap_model.version, '2023')
        self.assertFalse(clap_model.use_cuda) # Deve essere False per CPU
        self.assertEqual(clap_model.am_path, "/tmp/CLAP_weights_2023.pth")
        self.assertEqual(clap_model.lm_path, "/tmp/roberta-base")
        
        # 3. Verifica che 'audio_embedding_func' sia un Callable e sia il metodo corretto del mock
        self.assertTrue(callable(audio_embedding_func))
        self.assertEqual(audio_embedding_func.__name__, 'get_audio_embedding')

        # 4. Verifica che 'original_parameters' sia una lista di parametri (mockati)
        self.assertIsInstance(original_parameters, list)
        self.assertIsInstance(original_parameters[0], torch.nn.Parameter)

    @unittest.skipUnless(torch.cuda.is_available(), "Ignora test CUDA se la GPU non è disponibile")
    @patch('os.path.exists', return_value=True)
    @patch('os.getenv', side_effect=mock_getenv_clap_paths)
    @patch('src.models.CLAP', new=MockCLAP)
    def test_clap_initializer_success_cuda(self, mock_getenv, mock_exists):
        """Testa l'inizializzazione di CLAP su CUDA (se disponibile) con percorsi corretti (mockati)."""
        
        clap_model, audio_embedding_func, original_parameters = CLAP_initializer(device='cuda', use_cuda=True)
        
        self.assertIsNotNone(clap_model)
        self.assertIsNotNone(audio_embedding_func)
        self.assertIsNotNone(original_parameters)

        self.assertIsInstance(clap_model, MockCLAP)
        self.assertTrue(clap_model.use_cuda) # Deve essere True per CUDA
        self.assertEqual(clap_model.am_path, "/tmp/CLAP_weights_2023.pth")
        self.assertEqual(clap_model.lm_path, "/tmp/roberta-base")
        
        self.assertEqual(audio_embedding_func.__name__, 'get_audio_embedding')

    @patch('os.path.exists', side_effect=lambda path: path != "/tmp/CLAP_weights_2023.pth") # Solo i pesi non esistono
    @patch('os.getenv', side_effect=mock_getenv_clap_paths)
    def test_clap_initializer_audio_weights_not_found(self, mock_getenv, mock_exists):
        """Testa l'errore se il file dei pesi audio CLAP non viene trovato."""
        
        with self.assertRaises(FileNotFoundError) as context:
            CLAP_initializer(device='cpu', use_cuda=False)
            
        self.assertIn("Impossibile trovare i pesi CLAP a: /tmp/CLAP_weights_2023.pth", str(context.exception))

    @patch('os.path.exists', side_effect=lambda path: path != "/tmp/roberta-base") # Solo l'encoder text non esiste
    @patch('os.getenv', side_effect=mock_getenv_clap_paths)
    def test_clap_initializer_text_encoder_not_found(self, mock_getenv, mock_exists):
        """Testa l'errore se la directory dell'encoder testuale CLAP non viene trovata."""
        
        with self.assertRaises(FileNotFoundError) as context:
            CLAP_initializer(device='cpu', use_cuda=False)
            
        self.assertIn("Impossibile trovare l'encoder testuale CLAP a: /tmp/roberta-base", str(context.exception))

    @patch('os.getenv', side_effect=lambda key: None) # Tutte le variabili d'ambiente non impostate
    def test_clap_initializer_env_vars_missing(self, mock_getenv):
        """Testa l'errore se le variabili d'ambiente non sono impostate."""
        
        with self.assertRaises(ValueError) as context:
            CLAP_initializer(device='cpu', use_cuda=False)
            
        # Verifica che il messaggio di errore faccia riferimento alla prima variabile mancante
        self.assertIn("Variabile d'ambiente LOCAL_CLAP_WEIGHTS_PATH non impostata.", str(context.exception))
        
    @patch('os.path.exists', return_value=True)
    @patch('os.getenv', side_effect=mock_getenv_clap_paths)
    @patch('src.models.CLAP', side_effect=Exception("Simulazione errore CLAP init"))
    def test_clap_initializer_runtime_error(self, mock_CLAP, mock_getenv, mock_exists):
        """Testa l'errore runtime durante l'inizializzazione di msclap.CLAP."""
        
        with self.assertRaises(RuntimeError) as context:
            CLAP_initializer(device='cpu', use_cuda=False)
            
        self.assertIn("Errore durante l'inizializzazione del modello CLAP", str(context.exception))


    # --------------------------------------------------------------------------
    # 2.2 Test FinetunedModel (Già definiti, qui solo per completezza)
    # --------------------------------------------------------------------------
    
    def test_finetuned_model_initialization(self):
        """Testa l'inizializzazione del modello FinetunedModel."""
        classes = ["musicale", "ambientale", "speech"]
        model = FinetunedModel(classes, device='cpu')
        self.assertIsInstance(model, torch.nn.Module)
        self.assertIsInstance(model.classifier, torch.nn.Linear)
        self.assertEqual(model.classifier.in_features, 1024) 
        self.assertEqual(model.classifier.out_features, len(classes))
        self.assertEqual(model.device, 'cpu')
        self.assertEqual(model.classifier.weight.device.type, 'cpu')

    def test_finetuned_model_forward_numpy_input(self):
        """Testa la propagazione forward con input NumPy Array."""
        classes = ["A", "B"]
        model = FinetunedModel(classes, device='cpu')
        np_input = np.random.rand(2, 1024).astype(np.float32)
        output = model(np_input)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, torch.Size([2, len(classes)]))
        self.assertEqual(output.device.type, 'cpu')

    def test_finetuned_model_forward_tensor_input(self):
        """Testa la propagazione forward con input Tensor."""
        classes = ["A", "B", "C"]
        model = FinetunedModel(classes, device='cpu')
        tensor_input = torch.randn(1, 1024).to(model.classifier.weight.dtype)
        output = model(tensor_input)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, torch.Size([1, len(classes)]))

    def test_finetuned_model_forward_extra_dimension(self):
        """Testa la propagazione forward con un input a 3 dimensioni ([batch, 1, emb_dim])."""
        classes = ["X", "Y"]
        model = FinetunedModel(classes, device='cpu')
        tensor_input = torch.randn(4, 1, 1024).to(model.classifier.weight.dtype)
        output = model(tensor_input)
        self.assertEqual(output.shape, torch.Size([4, len(classes)]))
        self.assertEqual(output.dim(), 2)


    # --------------------------------------------------------------------------
    # 2.3 Test spectrogram_n_octaveband_generator (con mock di librosa)
    # --------------------------------------------------------------------------
    
    @patch('src.models.librosa.stft', return_value=np.random.rand(513, 200) + 1j * np.random.rand(513, 200)) # Mock STFT
    @patch('src.models.librosa.filters.cqm_to_note_filters', return_value=np.random.rand(5, 513)) # Mock CQT filters
    def test_spectrogram_generator_output_shape(self, mock_filters, mock_stft):
        """Testa la forma dell'output dello spettrogramma."""
        audio_data = np.random.rand(52100 * 3)
        n_octave = 5 
        center_freqs = np.array([100.0, 500.0, 2000.0, 4000.0, 8000.0])
        sampling_rate = 52100
        n_fft = 1024
        
        spectrogram_result = spectrogram_n_octaveband_generator(
            audio_data, n_octave, center_freqs, sampling_rate, n_fft
        )
        self.assertIsInstance(spectrogram_result, np.ndarray)
        self.assertEqual(spectrogram_result.shape, (5, 200))
        mock_filters.assert_called_once()
        mock_stft.assert_called_once()

    def test_spectrogram_generator_n_octave_mismatch(self):
        """Testa l'errore se n_octave non corrisponde al numero di center_freqs."""
        audio_data = np.random.rand(52100)
        n_octave = 5 
        center_freqs = np.array([100.0, 500.0, 2000.0, 4000.0]) # 4 frequenze
        sampling_rate = 52100
        n_fft = 1024
        
        with self.assertRaises(ValueError) as context:
            spectrogram_n_octaveband_generator(
                audio_data, n_octave, center_freqs, sampling_rate, n_fft
            )
            
        self.assertIn("Il numero di bande d'ottava (n_octave) deve corrispondere alla lunghezza di center_freqs", str(context.exception))


if __name__ == '__main__':
    # Modifica il percorso di sistema per trovare i moduli, assumendo una struttura src/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)

    unittest.main(argv=['first-arg-is-ignored'], exit=False)

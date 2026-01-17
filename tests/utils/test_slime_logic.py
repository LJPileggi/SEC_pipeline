import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Priorità assoluta moduli locali
sys.path.insert(0, os.getcwd())

from src.explainability.SLIME import SLIME
from src.models import CLAP_initializer, FinetunedModel

def generate_fourier_signal(duration, sr, freq=440):
    """
    Genera un segnale strutturato tramite serie di Fourier.
    Creato per indurre picchi spettrali rilevabili da CLAP.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Fondamentale + Armonica
    signal = 0.7 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    return torch.from_numpy(signal).float()

def test_slime_logic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_classes = ['Music', 'Voices', 'Birds']
    weights_path = os.getenv("TEST_WEIGHTS_PATH", "dummy_weights.pt")

    print("🧪 Testing SLIME Core Logic with High-Sensitivity setup...")

    # 1. SETUP CLASSIFICATORE AD ALTA SENSIBILITÀ
    # 🎯 FIX: Usiamo 1.0 per assicurarci che ogni minima variazione dell'embedding sposti i logit
    model = FinetunedModel(classes=dummy_classes, device=device)
    for param in model.parameters():
        with torch.no_grad():
            param.fill_(1.0) 
            
    if not os.path.exists(os.path.dirname(weights_path)):
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    classifier = FinetunedModel(classes=dummy_classes, device=device, weights_path=weights_path)

    # 2. SETUP CLAP
    _, audio_embedding, _ = CLAP_initializer(device=device, use_cuda=torch.cuda.is_available())

    # 3. GENERAZIONE DATI (Volume alzato e Segnale localizzato)
    sr = 51200
    duration = 1.0
    audio = torch.zeros(int(duration * sr)).to(device)
    
    # 🎯 FIX: Volume a 50.0 per rendere la rimozione di T0 "traumatica" per CLAP
    fourier_segment = generate_fourier_signal(0.1, sr).to(device)
    audio[:len(fourier_segment)] = fourier_segment * 50.0 
    
    # Spettrogramma coerente con energia massiccia all'inizio
    spec_linear = torch.zeros(1, 1, 27, 256).to(device)
    spec_linear[:, :, :10, :25] = 100.0 

    # 4. TEST SLIME: TEMPORAL EXPLANATIONS
    print("🧪 Testing SLIME: Time-based Explanations...")
    # 🎯 Aumentiamo n_samples a 100 per una regressione Ridge più stabile
    slime_time = SLIME(classifier, audio_embedding, explainer_type='time', n_samples=100)
    explanation_time = slime_time.explain_instance(audio, spec_linear, sr, class_idx=0)
    
    weights_time = list(explanation_time.values())
    avg_weight_time = np.mean(np.abs(weights_time))
    
    print(f"  -> Avg Weight (Time): {avg_weight_time:.10f}")
    
    # Se CLAP varia (come visto nei debug), i logit varieranno e Ridge avrà coefficienti != 0
    assert avg_weight_time > 1e-9, "I pesi temporali sono ancora zero! Verifica la varianza dei logit."
    assert abs(explanation_time['T0']) > abs(explanation_time['T1']), "SLIME non identifica T0 come segmento chiave"
    print("✅ Time-based explanation verified.")

    # 5. TEST SLIME: TIME-FREQUENCY EXPLANATIONS
    print("🧪 Testing SLIME: Time-Frequency-based Explanations...")
    slime_tf = SLIME(classifier, audio_embedding, explainer_type='time_frequency', n_samples=100)
    explanation_tf = slime_tf.explain_instance(audio, spec_linear, sr, class_idx=0)
    
    weights_tf = list(explanation_tf.values())
    avg_weight_tf = np.mean(np.abs(weights_tf))
    
    print(f"  -> Avg Weight (TF): {avg_weight_tf:.10f}")
    
    assert avg_weight_tf > 1e-9, "I pesi TF sono ancora zero!"
    # B0 deve essere dominante
    assert abs(explanation_tf['B0']) > abs(explanation_tf['B23']), "SLIME non identifica B0 come blocco chiave"
    print("✅ Time-Frequency explanation verified.")

if __name__ == "__main__":
    try:
        test_slime_logic()
        print("\n✨ SLIME CORE LOGIC VERIFIED SUCCESSFULLY ✨")
    except Exception as e:
        print(f"\n❌ TEST FAILED")
        import traceback
        traceback.print_exc()
        sys.exit(1)

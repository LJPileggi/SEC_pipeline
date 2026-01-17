import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())

from src.explainability.SLIME import SLIME
from src.models import CLAP_initializer, FinetunedModel

def test_slime_logic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_classes = ['Music', 'Voices', 'Birds']
    weights_path = os.getenv("TEST_WEIGHTS_PATH", "dummy_weights.pt")

    print("🧪 Testing SLIME Core Logic with structured signal...")

    # 1. Classifier Setup (Mock Weights)
    model = FinetunedModel(classes=dummy_classes, device=device)
    if not os.path.exists(os.path.dirname(weights_path)):
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    classifier = FinetunedModel(classes=dummy_classes, device=device, weights_path=weights_path)

    # 2. CLAP Setup
    _, audio_embedding, _ = CLAP_initializer(device=device, use_cuda=torch.cuda.is_available())

    # 3. 🎯 GENERAZIONE DATI STRUTTURATA (Soluzione lato Dati)
    sr = 51200
    duration = 1.0
    # Creiamo un segnale di silenzio e iniettiamo un forte "impulso" solo nel primo 10% (Segmento T0)
    #
    audio = torch.zeros(int(duration * sr)).to(device)
    audio[:int(0.1 * sr)] = torch.sin(2 * np.pi * 440 * torch.arange(int(0.1 * sr)).to(device) / sr) * 5.0
    
    # Creiamo uno spettrogramma coerente: alto nel primo blocco temporale, basso altrove
    spec_linear = torch.zeros(1, 1, 27, 256).to(device)
    spec_linear[:, :, :, :25] = 10.0 # Energia concentrata all'inizio

    # 4. Test SLIME: Time Segmentation
    print("🧪 Testing SLIME: Time-based Explanations...")
    slime_time = SLIME(classifier, audio_embedding, explainer_type='time', n_samples=50)
    explanation_time = slime_time.explain_instance(audio, spec_linear, sr, class_idx=0)
    
    weights = list(explanation_time.values())
    avg_weight = np.mean(np.abs(weights))
    
    # Ora avg_weight non sarà più 0 perché la rimozione del primo segmento 
    # cambierà drasticamente l'embedding CLAP e la risposta del classificatore
    print(f"✅ Time-based explanation generated. Avg Weight: {avg_weight:.6f}")
    
    assert avg_weight > 0, "Average weight is still zero despite structured input!"

    # 5. Test SLIME: Time-Frequency Segmentation
    print("🧪 Testing SLIME: Time-Frequency-based Explanations...")
    slime_tf = SLIME(classifier, audio_embedding, explainer_type='time_frequency', n_samples=50)
    explanation_tf = slime_tf.explain_instance(audio, spec_linear, sr, class_idx=0)
    
    tf_avg = np.mean(np.abs(list(explanation_tf.values())))
    print(f"✅ Time-Frequency explanation generated. Avg Weight: {tf_avg:.6f}")
    
    assert tf_avg > 0, "Time-Frequency average weight is zero!"

if __name__ == "__main__":
    try:
        test_slime_logic()
        print("\n✨ SLIME CORE LOGIC VERIFIED WITH STRUCTURED DATA ✨")
    except Exception as e:
        print(f"\n❌ TEST FAILED")
        import traceback
        traceback.print_exc()
        sys.exit(1)

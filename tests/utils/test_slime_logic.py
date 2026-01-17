import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())

from src.explainability.SLIME import SLIME
from src.models import CLAP_initializer, FinetunedModel

def generate_extreme_signal(duration, sr):
    """Genera un segnale a larga banda (multi-armonica) per saturare CLAP."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Somma di molte armoniche per creare un segnale molto ricco
    signal = sum([np.sin(2 * np.pi * (440 * i) * t) for i in range(1, 5)])
    return torch.from_numpy(signal).float()

def test_slime_logic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_classes = ['Target', 'Noise1', 'Noise2']
    weights_path = os.getenv("TEST_WEIGHTS_PATH", "dummy_weights.pt")

    print("🧪 Training Mock Classifier and Testing SLIME with Extreme Signal...")

    # 1. SETUP MODELLI
    _, audio_embedding, _ = CLAP_initializer(device=device, use_cuda=torch.cuda.is_available())
    classifier = FinetunedModel(classes=dummy_classes, device=device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    # 2. 🎯 ALLENAMENTO VELOCE (Overfitting sul segnale Target)
    print("  - Training classifier to recognize the extreme signal...")
    sr = 51200
    target_audio = torch.zeros(int(1.0 * sr)).to(device)
    target_audio[:int(0.1 * sr)] = generate_extreme_signal(0.1, sr).to(device) * 100.0
    
    # Alleniamo per 50 step per far sì che l'embedding di questo audio dia 'Target'
    classifier.train()
    for _ in range(50):
        optimizer.zero_grad()
        with torch.no_grad():
            output = audio_embedding(target_audio.unsqueeze(0))
            h_target = output[0][0] if isinstance(output, (tuple, list)) else output
            if h_target.dim() == 1: h_target = h_target.unsqueeze(0)
            
        pred = classifier(h_target)
        loss = criterion(pred, torch.tensor([0]).to(device))
        loss.backward()
        optimizer.step()
    
    classifier.eval()
    torch.save(classifier.state_dict(), weights_path)
    print(f"  - Training complete. Final Loss: {loss.item():.6f}")

    # 3. PREPARAZIONE INPUT PER SLIME
    spec_linear = torch.zeros(1, 1, 27, 256).to(device)
    spec_linear[:, :, :, :25] = 1000.0 # Valore enorme per lo spettrogramma HDF5

    # 4. TEST SLIME: TEMPORAL EXPLANATIONS
    print("🧪 Testing SLIME: Time-based Explanations...")
    slime_time = SLIME(classifier, audio_embedding, explainer_type='time', n_samples=100)
    explanation_time = slime_time.explain_instance(target_audio, spec_linear, sr, class_idx=0)
    
    weights_time = list(explanation_time.values())
    avg_weight = np.mean(np.abs(weights_time))
    
    print(f"  -> Avg Weight (Time): {avg_weight:.10f}")
    print(f"  -> T0 Weight (Signal): {explanation_time['T0']:.10f}")
    
    assert avg_weight > 1e-10, "I pesi sono ancora zero! La varianza delle predizioni è nulla."
    assert abs(explanation_time['T0']) > abs(explanation_time['T1']), "T0 dovrebbe dominare."
    print("✅ Time-based explanation verified.")

if __name__ == "__main__":
    try:
        test_slime_logic()
        print("\n✨ SLIME CORE LOGIC VERIFIED WITH TRAINED MODEL ✨")
    except Exception as e:
        print(f"\n❌ TEST FAILED")
        import traceback
        traceback.print_exc()
        sys.exit(1)

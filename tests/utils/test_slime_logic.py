import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
import gc

# --- CLAP MONKEY PATCH ---
import huggingface_hub
import transformers
import msclap

def universal_path_redirect(*args, **kwargs):
    w = os.getenv("LOCAL_CLAP_WEIGHTS_PATH")
    t = os.getenv("CLAP_TEXT_ENCODER_PATH")
    if any(x for x in args if 'msclap' in str(x)) or 'CLAP_weights' in str(kwargs): return w
    filename = kwargs.get('filename') or (args[1] if len(args) > 1 else None)
    return os.path.join(t, str(filename)) if (filename and t) else t

huggingface_hub.hf_hub_download = universal_path_redirect
transformers.utils.hub.cached_file = universal_path_redirect
msclap.CLAPWrapper.hf_hub_download = universal_path_redirect

sys.path.insert(0, os.getcwd())
from src.explainability.SLIME import SLIME
from src.models import CLAP_initializer, FinetunedModel

def generate_extreme_signal(duration, sr):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = sum([np.sin(2 * np.pi * (440 * i) * t) for i in range(1, 10)]) # Più armoniche
    return torch.from_numpy(signal).float()

def test_slime_logic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_classes = ['Target', 'Noise1', 'Noise2']
    weights_path = os.getenv("TEST_WEIGHTS_PATH", "dummy_weights.pt")

    print("🧪 Training Mock Classifier and Testing SLIME with Memory Management...")

    # 1. SETUP MODELLI
    _, audio_embedding, _ = CLAP_initializer(device=device, use_cuda=torch.cuda.is_available())
    classifier = FinetunedModel(classes=dummy_classes, device=device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()

    # 2. TRAINING
    sr = 51200
    target_audio = torch.zeros(int(1.0 * sr)).to(device)
    target_audio[:int(0.1 * sr)] = generate_extreme_signal(0.1, sr).to(device) * 50.0
    
    classifier.train()
    for _ in range(30): # Ridotte iterazioni per risparmiare RAM
        optimizer.zero_grad()
        with torch.no_grad():
            output = audio_embedding(target_audio.unsqueeze(0))
            h_target = output[0][0] if isinstance(output, (tuple, list)) else output
            if h_target.dim() == 1: h_target = h_target.unsqueeze(0)
            
        pred = classifier(h_target)
        loss = criterion(pred, torch.tensor([0]).to(device))
        loss.backward()
        optimizer.step()
    
    # 🎯 CRITICAL CLEANUP
    final_loss = loss.item()
    del optimizer
    del loss
    gc.collect()
    torch.cuda.empty_cache()
    classifier.eval()
    torch.save(classifier.state_dict(), weights_path)
    print(f"  - Training complete. Final Loss: {final_loss:.6f}")

    # 3. PREPARAZIONE INPUT
    spec_linear = torch.zeros(1, 1, 27, 256).to(device)
    spec_linear[:, :, :, :25] = 500.0

    # 4. EXPLANATION
    print("🧪 Testing SLIME: Time-based Explanations...")
    # Usiamo 50 campioni per bilanciare stabilità e memoria
    slime_time = SLIME(classifier, audio_embedding, explainer_type='time', n_samples=50)
    explanation_time = slime_time.explain_instance(target_audio, spec_linear, sr, class_idx=0)
    
    avg_weight = np.mean(np.abs(list(explanation_time.values())))
    print(f"  -> Avg Weight (Time): {avg_weight:.10f}")
    
    assert avg_weight > 1e-11, "I pesi sono zero! Varianza nulla."
    assert abs(explanation_time['T0']) > abs(explanation_time['T1']), "T0 (segnale) dovrebbe dominare."
    print("✅ SLIME core logic verified.")

if __name__ == "__main__":
    try:
        test_slime_logic()
        print("\n✨ ALL TESTS PASSED ✨")
    except Exception as e:
        print(f"\n❌ TEST FAILED")
        import traceback
        traceback.print_exc()
        sys.exit(1)

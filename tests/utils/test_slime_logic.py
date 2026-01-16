# ./tests/utils/test_slime_logic.py
import torch
import numpy as np
import sys
import os

# Priorità assoluta moduli locali
sys.path.insert(0, os.getcwd())

from src.explainability.SLIME import SLIME
from src.models import CLAP_initializer, FinetunedModel

def test_slime_logic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_classes = ['Music', 'Voices', 'Birds']
    weights_path = os.getenv("TEST_WEIGHTS_PATH", "dummy_weights.pt")

    print("🧪 Testing SLIME Core Logic and Surrogate Training...")

    # 1. Setup Mock Classifier (Mantra: Coerenza con test_config)
    print("  - Setting up mock classifier...")
    model = FinetunedModel(classes=dummy_classes, device=device)
    if not os.path.exists(os.path.dirname(weights_path)):
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    classifier = FinetunedModel(classes=dummy_classes, device=device, weights_path=weights_path)

    # 2. Setup Mock CLAP
    print("  - Setting up mock CLAP Wrapper...")
    _, audio_embedding, _ = CLAP_initializer(device=device, use_cuda=torch.cuda.is_available())

    # 3. Create dummy inputs
    sr = 51200
    duration = 1.0 
    audio = torch.randn(int(duration * sr)).to(device)
    spec_linear = torch.abs(torch.randn(1, 1, 27, 256)).to(device)

    # 4. Test SLIME: Time Segmentation
    print("🧪 Testing SLIME: Time-based Explanations...")
    # Usiamo pochi campioni per velocità di test
    slime_time = SLIME(classifier, audio_embedding, explainer_type='time', n_samples=20)
    explanation_time = slime_time.explain_instance(audio, spec_linear, sr, class_idx=0)
    
    assert isinstance(explanation_time, dict), "Explanation should be a dictionary"
    assert len(explanation_time) == 10, f"Expected 10 segments, got {len(explanation_time)}"
    
    # Verifica che i pesi non siano tutti zero o NaN
    weights = list(explanation_time.values())
    assert all(np.isfinite(w) for w in weights), "Found NaN/Inf in Ridge coefficients"
    assert any(w != 0 for w in weights), "Surrogate model learned only zero weights!"
    print(f"✅ Time-based explanation generated. Avg Weight: {np.mean(np.abs(weights)):.6f}")

    # 5. Test SLIME: Time-Frequency Segmentation
    print("🧪 Testing SLIME: Time-Frequency-based Explanations...")
    slime_tf = SLIME(classifier, audio_embedding, explainer_type='time_frequency', n_samples=20)
    explanation_tf = slime_tf.explain_instance(audio, spec_linear, sr, class_idx=0)
    
    assert isinstance(explanation_tf, dict), "Explanation should be a dictionary"
    # 4 bande frequenziali * 6 segmenti temporali = 24 blocchi
    assert len(explanation_tf) == 24, f"Expected 24 blocks, got {len(explanation_tf)}"
    assert all(np.isfinite(v) for v in explanation_tf.values()), "Found NaN/Inf in TF weights"
    print(f"✅ Time-Frequency explanation generated. Components: {len(explanation_tf)}")

if __name__ == "__main__":
    try:
        test_slime_logic()
        print("\n✨ SLIME CORE LOGIC VERIFIED SUCCESSFULLY ✨")
    except Exception as e:
        print(f"\n❌ TEST FAILED")
        import traceback
        traceback.print_exc()
        sys.exit(1)

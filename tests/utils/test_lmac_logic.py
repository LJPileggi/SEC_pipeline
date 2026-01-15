# ./tests/utils/test_lmac_logic.py
import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())

from src.explainability.LMAC import LMAC, Decoder
from src.models import CLAP_initializer, FinetunedModel

def test_full_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_classes = ['dog', 'cat', 'bird']
    weights_path = os.getenv("TEST_WEIGHTS_PATH")

    # 1. TEST: FinetunedModel Loading Logic
    print("🧪 Testing FinetunedModel weight loading...")
    # Create a dummy state dict to test loading
    temp_model = FinetunedModel(classes=dummy_classes, device=device)
    torch.save(temp_model.state_dict(), weights_path)
    
    classifier = FinetunedModel(classes=dummy_classes, device=device, weights_path=weights_path)
    print("✅ Classifier loaded successfully.")

    # 2. TEST: Decoder Mask Generation
    print("🧪 Testing Decoder mask generation...")
    spec_shape = (27, 256) # standard n-octave band shape
    decoder = Decoder(latent_dim_input=1024, output_spectrogram_shape=spec_shape).to(device)
    
    dummy_h = torch.randn(1, 1024).to(device)
    mask = decoder(dummy_h)
    
    assert mask.shape[-2:] == spec_shape, f"Wrong mask shape: {mask.shape}"
    assert 0.0 <= mask.min() and mask.max() <= 1.0, "Mask values out of [0, 1] range"
    print(f"✅ Decoder mask shape {mask.shape} is correct.")

    # 3. TEST: LMAC Masking Loss & Audio Loopback
    print("🧪 Testing LMAC Loss and Audio Reconstruction (Griffin-Lim)...")
    clap_wrapper, _, _ = CLAP_initializer(device=device, use_cuda=torch.cuda.is_available())
    lmac = LMAC(classifier=classifier, decoder=decoder, clap_model=clap_wrapper)
    
    # Simulate data from HDF5
    dummy_spec_linear = torch.abs(torch.randn(1, 1, 27, 256)).to(device)
    
    # Functional test of the loss calculation
    # This involves _apply_mask_to_audio and CLAP re-embedding
    loss, generated_mask = lmac.calculate_masking_loss(
        h_original=dummy_h, 
        linear_spec_X=dummy_spec_linear, 
        sampling_rate=51200
    )
    
    assert not torch.isnan(loss), "Loss is NaN!"
    print(f"✅ Masking loss calculated: {loss.item():.4f}")

    # 4. TEST: Final Interpretation Generation
    print("🧪 Testing generate_listenable_interpretation...")
    waveform = lmac.generate_listenable_interpretation(
        M=generated_mask, 
        linear_spec_X=dummy_spec_linear, 
        sampling_rate=51200
    )
    
    assert waveform.dim() == 2, "Reconstructed waveform should be (Batch, Samples)"
    print(f"✅ Listen-ready waveform generated with shape: {waveform.shape}")

if __name__ == "__main__":
    try:
        test_full_pipeline()
        print("\n✨ ALL CORE FUNCTIONALITIES VERIFIED SUCCESSFULLY ✨")
    except Exception as e:
        print(f"\n❌ TEST FAILED")
        import traceback
        traceback.print_exc()
        sys.exit(1)

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# --- CLAP MONKEY PATCH ---
import huggingface_hub
import transformers
import msclap

def universal_path_redirect(*args, **kwargs):
    weights_path = os.getenv("LOCAL_CLAP_WEIGHTS_PATH")
    text_path = os.getenv("CLAP_TEXT_ENCODER_PATH")
    if any(x for x in args if 'msclap' in str(x)) or 'CLAP_weights' in str(kwargs):
        return weights_path
    filename = kwargs.get('filename') or (args[1] if len(args) > 1 else None)
    if filename and text_path:
        return os.path.join(text_path, str(filename))
    return text_path

huggingface_hub.hf_hub_download = universal_path_redirect
transformers.utils.hub.cached_file = universal_path_redirect
msclap.CLAPWrapper.hf_hub_download = universal_path_redirect
# -------------------------

sys.path.insert(0, os.getcwd())
from src.explainability.LMAC import LMAC, Decoder
from src.models import CLAP_initializer, FinetunedModel

def test_full_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_classes = ['dog', 'cat', 'bird']
    weights_path = os.getenv("TEST_WEIGHTS_PATH", "dummy_classifier.pt")

    # 1. TEST: FinetunedModel Loading Logic
    temp_model = FinetunedModel(classes=dummy_classes, device=device)
    torch.save(temp_model.state_dict(), weights_path)
    classifier = FinetunedModel(classes=dummy_classes, device=device, weights_path=weights_path)
    print("✅ Classifier loaded.")

    # 2. TEST: Decoder Mask Generation
    spec_shape = (27, 256)
    decoder = Decoder(latent_dim_input=1024, output_spectrogram_shape=spec_shape).to(device)
    dummy_h = torch.randn(1, 1024).to(device)
    mask = decoder(dummy_h)
    assert mask.shape[-2:] == spec_shape
    print("✅ Decoder verified.")

    # 3. TEST: LMAC initialization with Patched CLAP
    clap_wrapper, audio_embedding, _ = CLAP_initializer(device=device, use_cuda=torch.cuda.is_available())
    lmac = LMAC(classifier=classifier, decoder=decoder, audio_embedding=audio_embedding)
    
    dummy_spec_linear = torch.abs(torch.randn(1, 1, 27, 256)).to(device)
    loss, generated_mask = lmac.calculate_masking_loss(h_original=dummy_h, linear_spec_X=dummy_spec_linear, sampling_rate=51200)
    assert not torch.isnan(loss)
    print(f"✅ Masking loss: {loss.item():.4f}")

    # 4. TEST: Listen-ready waveform
    waveform = lmac.generate_listenable_interpretation(M=generated_mask, linear_spec_X=dummy_spec_linear, sampling_rate=51200)
    assert waveform.dim() == 2
    print("✅ Waveform generated.")

if __name__ == "__main__":
    test_full_pipeline()

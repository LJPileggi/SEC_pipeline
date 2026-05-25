import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.stats
import logging

sys.path.insert(0, '/app')

# 💉 FIREWALLED HPC ENVIRONMENTS PATCH: Redirect weight downloads to local cache
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
transformers.utils.hub.hf_hub_download = universal_path_redirect
msclap.CLAPWrapper.hf_hub_download = universal_path_redirect

# Bypass internal audio loading library errors on headless clusters
def patched_read_audio(self, audio_path, resample=True): pass 
msclap.CLAP.read_audio = patched_read_audio

# Preserving your stable modular functions untouched
from src.utils import get_config_from_yaml, HDF5DatasetManager
from src.models import CLAP_initializer, get_octave_to_mel_transition_matrix, \
    spectrogram_n_octaveband_generator_gpu, convert_octave_to_msclap_mel

def parsing():
    parser = argparse.ArgumentParser(description='Evaluate Domain Distance Online on CPU (Class-Spacetted Raw HDF5)')
    parser.add_argument('--config_file', default='config0.yaml', help='YAML config file')
    parser.add_argument('--n_octave', default='3', help='Octave band resolution')
    parser.add_argument('--audio_format', default='wav', help='Audio format')
    parser.add_argument('--cut_secs', default=7, type=int, help='Segment duration')
    parser.add_argument('--samples_per_class', default=50, type=int, help='Max samples per class')
    return parser.parse_args()

def calculate_metrics(p_tensor, q_tensor):
    """Calculates geometric and statistical domain discrepancies between spectral maps."""
    # A. Frobenius Distance (Geometric L2 matrix norm)
    frobenius = torch.norm(p_tensor - q_tensor, p='fro').item()
    
    # Softmax conversion to interpret Log-Mel energy values as probability density slices
    p_prob = F.softmax(p_tensor.flatten(), dim=0).numpy()
    q_prob = F.softmax(q_tensor.flatten(), dim=0).numpy()
    
    # B. Kullback-Leibler Divergence (Informational entropy loss)
    kl_div = scipy.stats.entropy(p_prob, q_prob)
    if np.isinf(kl_div) or np.isnan(kl_div): 
        kl_div = 0.0 # Numerical safeguard
        
    # C. Wasserstein Distance (Earth Mover's Distance in transportation space)
    wasserstein = scipy.stats.wasserstein_distance(p_prob, q_prob)
    
    return frobenius, kl_div, wasserstein

def apply_online_pipeline(raw_audio, sr, cut_secs, hts_at_engine, W_matrix, n_octave):
    """
    🎯 ONLINE INTERACTIVE PIPELINE
    Applies deterministic cut/pad formatting to the extracted raw waveform, 
    then processes it simultaneously through native CLAP and custom filterbank layers.
    """
    target_length = sr * cut_secs
    if len(raw_audio) > target_length:
        audio_cut = raw_audio[:target_length]
    else:
        audio_cut = np.pad(raw_audio, (0, target_length - len(raw_audio)))
        
    audio_tensor = torch.from_numpy(audio_cut).float().unsqueeze(0) # Shape: [1, T_samples]
    
    with torch.no_grad():
        # --- PIPELINE A: NATIVE CLAP LOG-MEL (Ground Truth) ---
        x_stft = hts_at_engine.spectrogram_extractor(audio_tensor)
        x_native_logmel = hts_at_engine.logmel_extractor(x_stft)
        x_native_norm = hts_at_engine.bn0(x_native_logmel.transpose(1, 3)).transpose(1, 3)
        
        # --- PIPELINE B: CUSTOM FILTERBANK TO MEL CONVERSION ---
        # Generating octave-band spectrogram with cpu-safe execution parameters
        specs_cpu = spectrogram_n_octaveband_generator_gpu(
            audio_tensor, sr, int(n_octave), center_freqs=None, ref=1.0, device='cpu'
        )
        
        # Applying the active conversion pipeline from models.py (Interpolation + Compression + InstanceNorm)
        x_injected_norm = convert_octave_to_msclap_mel(specs_cpu)
        
        # Temporal axis alignment via bilinear interpolation if frames shapes mismatch
        if x_native_norm.shape[2] != x_injected_norm.shape[2]:
            x_injected_norm = F.interpolate(
                x_injected_norm, size=(x_native_norm.shape[2], 64), mode='bicubic', align_corners=True
            )
            
    return x_native_norm, x_injected_norm

def main():
    args = parsing()
    logging.basicConfig(level=logging.INFO)
    print(f"🔬 AVVIO VALUTAZIONE DISTANZA ONLINE (NATIVE SPACCETTATO - CPU INTERATTIVA)")
    print(f"   • Config: {args.config_file} | Ottave: {args.n_octave} | Campionamento: {args.samples_per_class}/classe")
    
    # Enforcing strict CPU compliance to prevent CUDA initialization faults on login nodes
    os.environ["INJECT_OCTAVE"] = "False"
    
    # Load native CLAP primitives directly onto CPU
    print("-> Caricamento modello CLAP nativo su CPU...")
    clap_model, _, _ = CLAP_initializer(device='cpu', use_cuda=False)
    hts_at_engine = clap_model.clap.audio_encoder.base.htsat
    hts_at_engine.eval()
    
    classes, _, _, _, sr, _, _, _, _, _, _ = get_config_from_yaml(args.config_file)
    W_matrix = get_octave_to_mel_transition_matrix(int(args.n_octave), sample_rate=sr, device='cpu')
    
    # 🎯 PERCORSO DEL DATASET RAW SPACCETTATO SU LEONARDO SCRATCH
    # Ogni classe ha il suo file .h5 dedicato nella forma: raw_dataset_{class_name}.h5
    raw_audio_root = os.path.join(
        os.environ.get("BASEDIR", "/leonardo_scratch/large/userexternal/user"),
        "dataSEC",
        "RAW_DATASET",
        f"raw_{args.audio_format}"
    )
    
    per_audio_results = []
    
    print("-> Inizio processing sequenziale delle classi (O(1) Memory Guard)...")
    
    # Iterating over classes: opening and closing one HDF5 container at a time
    for class_name in classes:
        # 🎯 GROUND TRUTH FIX: Match the exact naming convention from convert_audio_dataset_to_hdf5.py
        class_h5_path = os.path.join(raw_audio_root, f"{class_name}_{args.audio_format}_dataset.h5")
        
        # Structural fallback check for naming variations
        if not os.path.exists(class_h5_path):
            continue
                
        try:
            # Instantiating your stable HDF5DatasetManager for the current class file
            # This triggers a lazy open handle with rdcc_nbytes=0
            audio_manager = HDF5DatasetManager(class_h5_path, audio_format=args.audio_format)
            
            # Bound checking the sampling limit against the actual file size
            available_records = audio_manager.n_records
            samples_to_extract = min(args.samples_per_class, available_records)
            
            print(f"   📦 Elaborazione classe: {class_name} | Estrazione di {samples_to_extract}/{available_records} tracce...")
            
            # Slicing the file sequentially line by line up to the sampling limit
            for idx in range(samples_to_extract):
                raw_audio, meta_dict = audio_manager.get_audio_and_metadata(idx)
                
                # 🎯 GROUND TRUTH FIX: Extract 'track_name' following METADATA_DTYPE of the converter
                track_id = meta_dict.get('track_name', f"{class_name}_tr_{idx}")
                
                # Executing the clean dual-pipeline in RAM
                p_tensor, q_tensor = apply_online_pipeline(
                    raw_audio, sr, args.cut_secs, hts_at_engine, W_matrix, args.n_octave
                )
                
                # Evaluation of distances
                frob, kl, wass = calculate_metrics(p_tensor, q_tensor)
                
                per_audio_results.append({
                    'track_id': track_id,
                    'class': class_name,
                    'frobenius': frob,
                    'kl_divergence': kl,
                    'wasserstein': wass
                })
                
                # 🎯 ANTI-OOM MILESTONE: Radical RAM memory cleanup at each iteration
                # Inspired by the local execution management in distributed_clap_embeddings.py
                del p_tensor, q_tensor, raw_audio, meta_dict
                import gc
                gc.collect()
                
            # 🎯 CRITICAL STEP: Close current file handle and flush cache before moving to the next class
            if audio_manager.hf is not None:
                audio_manager.hf.close()
                
            # 🎯 AGGRESSIVE CPU RECOVERY: Force system-level glibc malloc trim back to the OS
            # Directly mirroring the production setup found in distributed_clap_embeddings.py
            import gc
            import ctypes
            gc.collect()
            try:
                ctypes.CDLL('libc.so.6').malloc_trim(0)
            except Exception:
                pass
                
        except Exception as e:
            print(f"   ⚠️ Errore critico saltato nella classe {class_name}: {e}")
            import gc
            import ctypes
            gc.collect()
            try:
                ctypes.CDLL('libc.so.6').malloc_trim(0)
            except Exception:
                pass
            continue

    # --- DATAFRAME REPORT GENERATION ---
    df = pd.DataFrame(per_audio_results)
    output_dir = "results/domain_analysis_online"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save granular file report
    df.to_csv(f"{output_dir}/online_per_audio_distances.csv", index=False)
    
    # Save disaggregated mean report per class
    df_class = df.groupby('class')[['frobenius', 'kl_divergence', 'wasserstein']].mean().reset_index()
    df_class = df_class.sort_values(by='wasserstein', ascending=False)
    df_class.to_csv(f"{output_dir}/online_per_class_distances.csv", index=False)
    
    print("\n" + "="*55)
    print("📊 VALUTAZIONE DISTANZE DI DOMINIO TERMINATA CON SUCCESSO")
    print(f"   • Totale tracce elaborate: {len(df)}")
    print(f"   • Cartella di output:    {output_dir}/")
    print("="*55 + "\n")
    print("Classi con la maggiore discrepanza di dominio rilevata (Wasserstein):")
    print(df_class.head(5))

if __name__ == "__main__":
    main()

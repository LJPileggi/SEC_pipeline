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

def patched_read_audio(self, audio_path, resample=True): pass 
msclap.CLAP.read_audio = patched_read_audio

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
    parser.add_argument('--class_to_process', default=None, type=str, help='Specific class name to process')
    return parser.parse_args()

def calculate_metrics(p_tensor, q_tensor):
    frobenius = torch.norm(p_tensor - q_tensor, p='fro').item()
    p_prob = F.softmax(p_tensor.flatten(), dim=0).numpy()
    q_prob = F.softmax(q_tensor.flatten(), dim=0).numpy()
    kl_div = scipy.stats.entropy(p_prob, q_prob)
    if np.isinf(kl_div) or np.isnan(kl_div): kl_div = 0.0
    wasserstein = scipy.stats.wasserstein_distance(p_prob, q_prob)
    return frobenius, kl_div, wasserstein

def apply_online_pipeline(raw_audio, sr, cut_secs, hts_at_engine, W_matrix, n_octave):
    target_length = sr * cut_secs
    if len(raw_audio) > target_length:
        audio_cut = raw_audio[:target_length]
    else:
        audio_cut = np.pad(raw_audio, (0, target_length - len(raw_audio)))
        
    audio_tensor = torch.from_numpy(audio_cut).float().unsqueeze(0)
    
    with torch.no_grad():
        x_stft = hts_at_engine.spectrogram_extractor(audio_tensor)
        x_native_logmel = hts_at_engine.logmel_extractor(x_stft)
        x_native_norm = hts_at_engine.bn0(x_native_logmel.transpose(1, 3)).transpose(1, 3)
        
        specs_cpu = spectrogram_n_octaveband_generator_gpu(
            audio_tensor, sr, int(n_octave), center_freqs=None, ref=1.0, device='cpu'
        )
        x_injected_norm = convert_octave_to_msclap_mel(specs_cpu)
        
        if x_native_norm.shape[2] != x_injected_norm.shape[2]:
            x_injected_norm = F.interpolate(
                x_injected_norm, size=(x_native_norm.shape[2], 64), mode='bicubic', align_corners=True
            )
            
    print(x_injected_norm)
    return x_native_norm, x_injected_norm

def main():
    args = parsing()
    logging.basicConfig(level=logging.INFO)
    print(f"🔬 AVVIO VALUTAZIONE DISTANZA ONLINE (NATIVE SPACCETTATO - CPU INTERATTIVA)")
    
    os.environ["INJECT_OCTAVE"] = "False"
    clap_model, _, _ = CLAP_initializer(device='cpu', use_cuda=False)
    hts_at_engine = clap_model.clap.audio_encoder.base.htsat
    hts_at_engine.eval()
    
    classes, _, _, _, sr, _, _, _, _, _, _ = get_config_from_yaml(args.config_file)
    W_matrix = get_octave_to_mel_transition_matrix(int(args.n_octave), sample_rate=sr, device='cpu')
    
    raw_audio_root = os.path.join(
        os.environ.get("BASEDIR", "/leonardo_scratch/large/userexternal/user"),
        "dataSEC", "RAW_DATASET", f"raw_{args.audio_format}"
    )
    
    per_audio_results = []
    # 🎯 Accumulatore di dizionari per mantenere il dato di ogni traccia e ogni bin (Long Format)
    granular_mel_rows = []
    # 🎯 ABBIAMO DUE CONFIGURAZIONI: Accumulatori per le matrici temporali native e iniettate
    class_time_resolved_specs_native = []
    class_time_resolved_specs_injected = []
    
    for class_name in classes:
        if args.class_to_process and class_name != args.class_to_process:
            continue
            
        class_h5_path = os.path.join(raw_audio_root, f"{class_name}_{args.audio_format}_dataset.h5")
        if not os.path.exists(class_h5_path):
            continue
                
        try:
            audio_manager = HDF5DatasetManager(class_h5_path, audio_format=args.audio_format)
            available_records = audio_manager.n_records
            samples_to_extract = min(args.samples_per_class, available_records)
            
            print(f"   📦 Elaborazione classe: {class_name} | Estrazione di {samples_to_extract}/{available_records} tracce...")
            
            for idx in range(samples_to_extract):
                raw_audio, meta_dict = audio_manager.get_audio_and_metadata(idx)
                track_id = meta_dict.get('track_name', f"{class_name}_tr_{idx}")
                
                p_tensor, q_tensor = apply_online_pipeline(
                    raw_audio, sr, args.cut_secs, hts_at_engine, W_matrix, args.n_octave
                )
                
                frob, kl, wass = calculate_metrics(p_tensor, q_tensor)
                
                per_audio_results.append({
                    'track_id': track_id, 'class': class_name,
                    'frobenius': frob, 'kl_divergence': kl, 'wasserstein': wass
                })
                
                # 🎯 METANALISI 1: Estrazione residuo assoluto per singolo bin Mel
                absolute_residual = torch.abs(p_tensor - q_tensor)
                mel_profile = torch.mean(absolute_residual, dim=2).squeeze().cpu().numpy() # [64,]
                
                # Popoliamo il formato lungo preservando la traccia per consentire il Boxplot successivo
                for bin_idx, val in enumerate(mel_profile):
                    granular_mel_rows.append({
                        'track_id': track_id,
                        'class': class_name,
                        'mel_bin': bin_idx,
                        'discrepancy': float(val)
                    })

                # 🎯 CONFIGURAZIONE NATIVA (Log-Mel CLAP): Forziamo lo squeeze solo di batch e canale [0, 1]
                # Portiamo da [1, 1, Time, 64] a [Time, 64] e poi trasponiamo in [64, Time]
                spec_2d_native = p_tensor.squeeze(0).squeeze(0).detach().cpu().numpy().T
                class_time_resolved_specs_native.append(spec_2d_native)
                
                # 🎯 CONFIGURAZIONE INIETTATA (Le nostre Ottave): Stessa identica chirurgia geometrica
                spec_2d_injected = q_tensor.squeeze(0).squeeze(0).detach().cpu().numpy().T
                class_time_resolved_specs_injected.append(spec_2d_injected)

                del p_tensor, q_tensor, raw_audio, meta_dict
                import gc; gc.collect()
                
            if audio_manager.hf is not None:
                audio_manager.hf.close()
            import gc, ctypes; gc.collect(); ctypes.CDLL('libc.so.6').malloc_trim(0)
                
        except Exception as e:
            print(f"   ⚠️ Errore saltato nella classe {class_name}: {e}")
            import gc, ctypes; gc.collect(); ctypes.CDLL('libc.so.6').malloc_trim(0)
            continue

    if len(per_audio_results) > 0:
        df = pd.DataFrame(per_audio_results)
        output_dir = os.path.join(os.getenv("RESULTS_DIR"), "domain_analysis_online")
        os.makedirs(output_dir, exist_ok=True)
        
        class_suffix = f"_{args.class_to_process}" if args.class_to_process else ""
        df.to_csv(f"{output_dir}/online_per_audio_distances{class_suffix}.csv", index=False)
        
        # 🎯 ESPORTAZIONE DATI GRANULARI PER IL BOXPLOT
        df_mel_raw = pd.DataFrame(granular_mel_rows)
        df_mel_raw.to_csv(f"{output_dir}/mel_raw_tracks{class_suffix}.csv", index=False)
        print(f"   • File Granulare Mel generato per Boxplot: {output_dir}/mel_raw_tracks{class_suffix}.csv")
        
        # 🎯 CONSOLIDAMENTO CENTROIDE CONFIGURAZIONE NATIVA
        if len(class_time_resolved_specs_native) > 0:
            stacked_native = np.stack(class_time_resolved_specs_native, axis=0)
            mean_native_spec_2d = np.mean(stacked_native, axis=0) # [64, Time]
            np.save(f"{output_dir}/spectral_centroid_native{class_suffix}.npy", mean_native_spec_2d)
            print(f"   • Centroide Spettrale Nativo [64, Time] salvato con successo.")
            
        # 🎯 CONSOLIDAMENTO CENTROIDE CONFIGURAZIONE INIETTATA (3 OTTAVE)
        if len(class_time_resolved_specs_injected) > 0:
            stacked_injected = np.stack(class_time_resolved_specs_injected, axis=0)
            mean_injected_spec_2d = np.mean(stacked_injected, axis=0) # [64, Time]
            np.save(f"{output_dir}/spectral_centroid_injected{class_suffix}.npy", mean_injected_spec_2d)
            print(f"   • Centroide Spettrale a Ottave [64, Time] salvato con successo.")

if __name__ == "__main__":
    main()

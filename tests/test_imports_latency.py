import time
import os
import sys

# Aggiungi il percorso del tuo codice, se necessario
# Assumiamo che la cartella 'src' sia accessibile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
print(f"Path di ricerca: {sys.path}")


def measure_import_latency(module_name):
    """Misura il tempo impiegato per importare un modulo."""
    start = time.perf_counter()
    try:
        __import__(module_name)
        end = time.perf_counter()
        return end - start
    except Exception as e:
        end = time.perf_counter()
        print(f"ERRORE nell'import di {module_name}: {e}")
        return end - start

def main():
    print("--- Inizio Misurazione Latenza Import ---")

    # Moduli base (Python/Standard)
    time_os = measure_import_latency('os')
    time_sys = measure_import_latency('sys')
    
    # Moduli scientifici e data science
    time_np = measure_import_latency('numpy')
    time_pd = measure_import_latency('pandas')
    time_yaml = measure_import_latency('yaml')
    time_glob = measure_import_latency('glob')
    time_h5py = measure_import_latency('h5py')
    
    # Moduli audio e Torch
    time_torch = measure_import_latency('torch')
    time_librosa = measure_import_latency('librosa')
    time_sf = measure_import_latency('soundfile')
    time_dist = measure_import_latency('torch.distributed')

    # I tuoi moduli (simuliamo l'importazione delle tue librerie principali)
    # NOTA: Queste misureranno il tempo necessario per risolvere l'importazione e
    # l'esecuzione di qualsiasi codice a livello di modulo (inizializzazioni)

    # Nota: L'importazione dei moduli locali DEVE avvenire DOPO aver gestito il sys.path.
    
    # Per includere i tuoi moduli come 'src.utils', usa la sintassi completa
    time_dirs_config = measure_import_latency('src.dirs_config')
    time_utils = measure_import_latency('src.utils')
    time_models = measure_import_latency('src.models')
    
    # Includi il modulo principale che coordina tutto
    time_dist_clap = measure_import_latency('src.distributed_clap_embeddings')


    print("\n--- Risultati Latenza Import (in secondi) ---")
    
    # Crea un elenco di risultati e stampalo
    results = {
        "os": time_os,
        "sys": time_sys,
        "numpy": time_np,
        "pandas": time_pd,
        "yaml": time_yaml,
        "glob": time_glob,
        "h5py": time_h5py,
        "torch": time_torch,
        "librosa": time_librosa,
        "soundfile": time_sf,
        "torch.distributed": time_dist,
        "src.dirs_config": time_dirs_config,
        "src.utils": time_utils,
        "src.models": time_models,
        "src.distributed_clap_embeddings": time_dist_clap,
    }
    
    total_import_time = 0
    
    for module, latency in results.items():
        print(f"{module:<35}: {latency:.4f} s")
        total_import_time += latency
        
    print("-" * 45)
    print(f"Tempo Totale per tutti gli Import: {total_import_time:.4f} s")
    print("---------------------------------------------------")


if __name__ == '__main__':
    main()

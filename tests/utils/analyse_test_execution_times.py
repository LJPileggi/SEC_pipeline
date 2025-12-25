import os
import argparse
import json
import re
from typing import Dict, Any, Tuple, Optional

# Aggiungi il percorso 'src' al path di sistema per le importazioni
import sys
sys.path.append('.')

config_test_folder = os.path.join('.', 'tmp_test_logs', 'PREPROCESSED_DATASET')

# Regex per estrarre cut_secs e classe da una chiave stringa come "(1, 'Music')"
TASK_KEY_PATTERN = re.compile(r"\((?P<cut_secs>\d+), '(?P<class_name>.*?)'\)")

def parse_task_key(key: str) -> Optional[Tuple[int, str]]:
    """Estrae cut_secs e nome della classe dalla chiave JSON."""
    match = TASK_KEY_PATTERN.match(key)
    if match:
        try:
            cut_secs = int(match.group('cut_secs'))
            class_name = match.group('class_name')
            return cut_secs, class_name
        except ValueError:
            return None
    return None


def analyze_execution_times(audio_format: str, n_octave: str, config_file: str) -> Dict[str, Any]:
    """
    Versione aggiornata: scansiona tutte le sottocartelle X_secs e aggrega i log.
    """
    # 1. Percorso base dove cercare le sottocartelle dei cut_secs
    base_folder = os.path.join(config_test_folder, f'{audio_format}', f'{n_octave}_octave')
    
    if not os.path.exists(base_folder):
        return {"error": f"Cartella base non trovata: {base_folder}"}

    # Inizializziamo contenitori per l'aggregazione
    all_log_data = {}
    
    # 2. Scansione ricorsiva per trovare tutti i log.json unificati
    found_logs = False
    for root, dirs, files in os.walk(base_folder):
        if "log.json" in files:
            log_file_path = os.path.join(root, "log.json")
            try:
                with open(log_file_path, 'r') as f:
                    data = json.load(f)
                    # Aggreghiamo i task saltando la chiave 'config'
                    for k, v in data.items():
                        if k != "config":
                            all_log_data[k] = v
                    found_logs = True
            except Exception as e:
                print(f"Avviso: Errore nella lettura di {log_file_path}: {e}", file=sys.stderr)

    if not found_logs:
        return {"error": f"Nessun file log.json trovato nelle sottocartelle di: {base_folder}"}

    # --- Da qui in poi la logica di calcolo rimane identica alla tua ---
    results: Dict[str, Dict[str, float]] = {}
    total_pipeline_time: float = 0.0
    
    for task_key, data in all_log_data.items():
        parsed_key = parse_task_key(task_key)
        if not parsed_key:
            continue
            
        process_times = data.get("process_time", [])
        n_embeddings = data.get("n_embeddings_per_run", [])
        
        total_time_task = sum(process_times)
        total_embeddings = sum(n_embeddings)
        
        avg_time_per_embedding = 0.0
        if total_embeddings > 0:
            avg_time_per_embedding = total_time_task / total_embeddings
        
        total_pipeline_time += total_time_task
        results[task_key] = {
            "total_time_seconds": total_time_task,
            "total_embeddings": total_embeddings,
            "avg_time_per_embedding_seconds": avg_time_per_embedding
        }

    analysis_results: Dict[str, Any] = {
        'Total_Cumulative_Worker_Time_seconds': total_pipeline_time,
        'Task_Metrics': results,
        'Cut_Secs_Groups': {}
    }
    
    for task_key, metrics in results.items():
        parsed = parse_task_key(task_key)
        if not parsed: continue
        cut_secs, class_name = parsed
        
        if cut_secs not in analysis_results['Cut_Secs_Groups']:
            analysis_results['Cut_Secs_Groups'][cut_secs] = []
            
        analysis_results['Cut_Secs_Groups'][cut_secs].append({
            "class": class_name, "metrics": metrics
        })
            
    return analysis_results

def print_analysis_results(results: Dict[str, Any]):
    """Formatta e stampa i risultati dell'analisi per la console."""
    if 'error' in results:
        print(f"\nERRORE Analisi Tempi: {results['error']}")
        return
        
    print("\n=============================================")
    print("      ANALISI TEMPI DI ESECUZIONE CLAP       ")
    print("=============================================")
    
    total_time = results.get('Total_Cumulative_Worker_Time_seconds', 0.0)
    print(f"Tempo Cumulativo Totale Worker: {total_time:.2f} secondi ({total_time / 60.0:.2f} minuti)")
    print("---------------------------------------------")

    if 'Task_Metrics' in results and results['Task_Metrics']:
        
        # Ordina le chiavi (cut_secs) e poi le classi al loro interno
        sorted_cut_secs = sorted(results['Cut_Secs_Groups'].keys())
        
        for cut_secs in sorted_cut_secs:
            print(f"\n[ RISULTATI PER: cut_secs = {cut_secs} secondi ]")
            
            tasks = results['Cut_Secs_Groups'][cut_secs]
            # Ordina per classe
            sorted_tasks = sorted(tasks, key=lambda x: x['class'])
            
            for task in sorted_tasks:
                metrics = task['metrics']
                
                # Tempo Totale per Configurazione (somma dei process_time)
                total_task_time = metrics['total_time_seconds']
                # Tempo Medio per Embedding
                avg_time = metrics['avg_time_per_embedding_seconds']
                # Embeddings totali
                total_emb = metrics['total_embeddings']
                
                print(f"  -- Classe: {task['class']} --")
                print(f"     Tempo Totale Config. (sec): {total_task_time:.2f}")
                print(f"     Embeddings Totali:          {int(total_emb)}")
                print(f"     Tempo Medio per Emb. (sec): {avg_time:.6f}") # Precisione più alta per il tempo per embedding
            
            print("---------------------------------------------")

    else:
        print("Nessun task con tempo di esecuzione rilevato nel log.json.")
        print("=============================================")

# (Parsing e blocco __main__ per il testing)
def parsing():
    parser = argparse.ArgumentParser(description='Analyze CLAP execution times from log file.')
    parser.add_argument('--config_file', dest='config_file', required=True,
            help='config file used for the execution.')
    parser.add_argument('--n_octave', dest='n_octave', required=True,
            help='octaveband split used for the execution.')
    parser.add_argument('--audio_format', dest='audio_format', required=True,
            help='audio format used for the execution.')
    return parser.parse_args()


if __name__ == '__main__':
    import shutil
    print("--- DEBUGGING LOCALE: Test del modulo di Analisi Tempi (JSON Unificato) ---")
    
    TEST_AUDIO_FORMAT = "wav"
    TEST_N_OCTAVE = "1"
    TEST_CONFIG_FILE = "test_config.yaml" 
    
    # 1. Imposta la cartella temporanea (mock) e pulisci
    mock_embed_folder = os.path.join(config_test_folder, TEST_AUDIO_FORMAT, f'{TEST_N_OCTAVE}_octave')
    if os.path.exists(mock_embed_folder): shutil.rmtree(mock_embed_folder)
    os.makedirs(mock_embed_folder, exist_ok=True)
        
    mock_log_file_path = os.path.join(mock_embed_folder, "log.json") 
    
    # 2. Genera contenuto fittizio del log (simulazione dell'output di join_logs)
    mock_log_content = {
        "config": {
            "config_file": TEST_CONFIG_FILE,
            "n_octave": TEST_N_OCTAVE,
            "audio_format": TEST_AUDIO_FORMAT
        },
        "(1, 'Music')": {
            "process_time": [10.5, 12.0], 
            "n_embeddings_per_run": [50, 50],
            "rank": [0, 1],
            "completed": True
        },
        "(2, 'Voices')": {
            "process_time": [15.2], 
            "n_embeddings_per_run": [100],
            "rank": [0],
            "completed": True
        },
        "(1, 'Birds')": {
            "process_time": [8.0, 7.5, 9.1], 
            "n_embeddings_per_run": [20, 20, 20],
            "rank": [0, 1, 2],
            "completed": True
        },
        "(2, 'Train')": {
            "process_time": [10.0, 11.0], 
            "n_embeddings_per_run": [50, 50],
            "rank": [0, 1],
            "completed": True
        }
    }
    
    with open(mock_log_file_path, 'w') as f:
        json.dump(mock_log_content, f, indent=4)
        
    print(f"File di log JSON fittizio generato in: {mock_log_file_path}")
    
    # 3. Esegui l'analisi
    results = analyze_execution_times(TEST_AUDIO_FORMAT, TEST_N_OCTAVE, TEST_CONFIG_FILE)
    
    # 4. Stampa i risultati
    print_analysis_results(results)
    shutil.rmtree(config_test_folder)

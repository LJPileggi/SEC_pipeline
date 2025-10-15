from pathlib import Path
import os
import sys

# SOSTITUISCI QUESTO CON IL TUO VERO PERCORSO RADICE
user = os.environ.get('USER')
BASE_DIR = Path(f'/leonardo_scratch/large/{user}/dataSEC/RAW_DATASET') 

AUDIO_FORMAT = 'wav'

if not BASE_DIR.is_dir():
    print(f"ERRORE: La directory di base non esiste o il percorso è sbagliato: {BASE_DIR}")
    sys.exit(1)

# Esegui la scansione e prendi i primi 5 file
found_files = list(BASE_DIR.rglob(f'*.{AUDIO_FORMAT}'))

if not found_files:
    print(f"ATTENZIONE: Nessun file '.{AUDIO_FORMAT}' trovato in {BASE_DIR} o nelle sue sottocartelle.")
else:
    print(f"Trovati {len(found_files)} file totali.")
    print("Primi 5 file trovati:")
    for f in found_files[:5]:
        print(f)

# Inserisci questo alla fine del tuo script .sh di esecuzione
echo "📊 Unione risultati in corso..."

# Definisci il path dei risultati (deve coincidere con validation_filepath)
RES_DIR="results/validation/${BENCHMARK_AUDIO_FORMAT}/${BENCHMARK_N_OCTAVE}"
MODEL_TYPE="linear"

python3 -c "
import os, json, glob, pandas as pd
target_dir = '$RES_DIR'
model_type = '$MODEL_TYPE'

# 1. Merge JSON
json_files = sorted(glob.glob(os.path.join(target_dir, f'validation_ms_results_{model_type}_rank_*.json')))
combined_json = {}
for f in json_files:
    with open(f, 'r') as j:
        data = json.load(j)
        for k, v in data.items():
            if k not in combined_json: combined_json[k] = []
            combined_json[k].extend(v)
with open(os.path.join(target_dir, 'FINAL_ms_results.json'), 'w') as f:
    json.dump(combined_json, f, indent=4)

# 2. Merge CSV
csv_files = sorted(glob.glob(os.path.join(target_dir, f'validation_ms_results_{model_type}_rank_*.csv')))
if csv_files:
    combined_df = pd.concat([pd.read_csv(f) for f in csv_files])
    combined_df = combined_df.sort_values('accuracy', ascending=False)
    combined_df.to_csv(os.path.join(target_dir, 'FINAL_ms_results.csv'), index=False)

print(f'✅ Consolidati {len(json_files)} file di rank.')
"

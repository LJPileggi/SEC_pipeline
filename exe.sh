#!/bin/bash

SCRIPT_TO_RUN="./scripts/"

echo "Seleziona lo script Python da eseguire:"
echo "1) get_clap_embeddings.py (richiede un nome e un'opzione verbosa)"
echo "2) classifier_finetuning.py (richiede un percorso e un'opzione di output)"
read -p "Inserisci il numero (1 o 2): " choice

case $choice in
    1)
        SCRIPT_TO_RUN+="get_clap_embeddings.py"
        ;;
    2)
        SCRIPT_TO_RUN+="classifier_finetuning.py"
        ;;
    *)
        echo "Scelta non valida. Uscita."
        exit 1
        ;;
esac

echo "Hai selezionato: $SCRIPT_TO_RUN"

COMMAND_ARGS=""
echo "Ora inserisci le flag e i loro valori. Digita 'done' per terminare."

while true; do
    read -p "Inserisci il nome della flag (es. --name, --path): " flag_name

    if [[ "$flag_name" == "done" ]]; then
        break
    fi

    if [[ "$flag_name" == "--verbose" || "$flag_name" == "-v" ]]; then
        COMMAND_ARGS+=" $flag_name"
    else
        read -p "Inserisci il valore per '$flag_name': " flag_value
        COMMAND_ARGS+=" $flag_name \"$flag_value\""
    fi
done

echo "Eseguo il comando: python3 $SCRIPT_TO_RUN$COMMAND_ARGS"
python3 "$SCRIPT_TO_RUN" $COMMAND_ARGS

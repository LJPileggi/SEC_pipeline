#!/bin/bash

SCRIPT_TO_RUN="./scripts/"

echo "Select python script:"
echo "1) get_clap_embeddings.py (args required: --config_file, --n_octave, --audio_format)"
echo "2) classifier_finetuning.py (args required: --config_file, --validation_filepath, --n_octave, --audio_format)"
echo "3) create_new_config.py"
read -p "Insert number (1, 2, 3): " choice

case $choice in
    1)
        SCRIPT_TO_RUN+="get_clap_embeddings.py"
        ;;
    2)
        SCRIPT_TO_RUN+="classifier_finetuning.py"
        ;;
    3)
        SCRIPT_TO_RUN+="create_new_config.py"
        ;;
    *)
        echo "Invalid script. Exit."
        exit 1
        ;;
esac

echo "Selected: $SCRIPT_TO_RUN"

COMMAND_ARGS=""
echo "Insert args with their values. Type 'done' to terminate."

while true; do
    read -p "Insert arg name (es. --name, --path): " flag_name

    if [[ "$flag_name" == "done" ]]; then
        break
    fi

    if [[ "$flag_name" == "--verbose" || "$flag_name" == "-v" ]]; then
        COMMAND_ARGS+=" $flag_name"
    fi
    if [[ "$flag_name" == "--help" || "$flag_name" == "-h" ]]; then
        COMMAND_ARGS+=" $flag_name"
    else
        read -p "Insert value for '$flag_name': " flag_value
        COMMAND_ARGS+=" $flag_name \"$flag_value\""
    fi
done

echo "Executing command: python3 $SCRIPT_TO_RUN$COMMAND_ARGS"
python3 "$SCRIPT_TO_RUN" $COMMAND_ARGS

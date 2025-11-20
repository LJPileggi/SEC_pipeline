# Definizione delle variabili principali per il lancio
USER_SCRATCH="/leonardo_scratch/large/userexternal/$USER"
PROJECT_ROOT_DIR="$USER_SCRATCH/SEC_pipeline" 
SIF_PATH="$PROJECT_ROOT_DIR/.containers/clap_pipeline.sif"

# La cartella che contiene i pesi e i dati
DATA_ROOT="$USER_SCRATCH/dataSEC"
WEIGHTS_DIR="$PROJECT_ROOT_DIR/.clap_weights"


# Comando di Esecuzione (singularity exec)
if [ "$1" == "test_latency.sh" ]; then
    singularity exec \
        --bind $DATA_ROOT:/app/data \
        --bind $WEIGHTS_DIR:/app/weights \
        $SIF_PATH \
        "./tests/$1"
else
    singularity exec \
        --bind $DATA_ROOT:/app/data \
        --bind $WEIGHTS_DIR:/app/weights \
        $SIF_PATH \
        python3 tests/"$1"
fi

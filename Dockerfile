# Usa la stessa base NVIDIA PyTorch
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Imposta variabili d'ambiente globali
ENV PATH="/usr/local/bin:${PATH}"
ENV HF_HOME="/tmp/.huggingface_cache"
ENV LC_ALL="C.UTF-8"
ENV LANG="C.UTF-8"
ENV CLAP_TEXT_ENCODER_PATH="/opt/models/roberta-base"

# Installazione dipendenze di sistema (equivalente a %post)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libsndfile1 \
    libsox-fmt-all \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Installazione dipendenze Python
RUN pip install --no-cache-dir --upgrade \
    pip tensorflow torch scikit-learn xgboost \
    transformers pandas matplotlib scipy msclap pydub \
    librosa soundfile pyyaml tqdm_multiprocess h5py

# Scarica e salva il Text Encoder (equivalente a %post script)
RUN mkdir -p /opt/models/roberta-base && \
    python3 -c "import transformers; \
    model_name = 'roberta-base'; \
    tokenizer = transformers.RobertaTokenizerFast.from_pretrained(model_name); \
    tokenizer.save_pretrained('/opt/models/roberta-base'); \
    model = transformers.RobertaModel.from_pretrained(model_name); \
    model.save_pretrained('/opt/models/roberta-base')"

# ENTRYPOINT/CMD non è necessario per un'immagine destinata all'HPC/Singularity.

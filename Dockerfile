FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install torchaudio==2.3.1
RUN pip install faster-whisper flask numpy pyannote.audio transformers
RUN pip install transformers -U
RUN pip install pydub
RUN pip install ctranslate2==4.4.0
RUN mkdir -p /app/models
COPY api.py /app/
COPY transcriber.py /app/
EXPOSE 4200
CMD ["python3", "api.py"]
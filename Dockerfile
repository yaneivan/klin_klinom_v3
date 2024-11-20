FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torchaudio==2.3.1
RUN pip install faster-whisper flask numpy pyannote.audio transformers

RUN pip install transformers -U

RUN pip install pytest

RUN mkdir -p /app/models
RUN mkdir -p /app/tests

COPY api.py /app/
COPY transcriber.py /app/
COPY tests/ /app/tests/

EXPOSE 4200

CMD ["python3", "api.py"]
FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04

WORKDIR /app

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install soundfile

RUN pip install torchaudio==2.3.1
RUN pip install faster-whisper flask numpy pyannote.audio transformers

RUN pip install transformers -U

RUN pip install pytest

RUN pip install pydub

RUN mkdir -p /app/models
RUN mkdir -p /app/tests
RUN mkdir -p /app/uploads

COPY api.py /app/
COPY transcriber.py /app/
COPY tests/ /app/tests/
COPY uploads/ /app/uploads/

EXPOSE 4200

CMD ["python3", "api.py"]
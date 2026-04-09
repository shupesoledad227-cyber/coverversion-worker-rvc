# RVC v2 Train + Infer Worker for RunPod Serverless
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install our core dependencies first
RUN pip install --no-cache-dir \
    runpod \
    requests \
    pedalboard \
    demucs \
    matplotlib \
    scipy \
    librosa \
    soundfile \
    praat-parselmouth \
    pyworld \
    faiss-cpu \
    torchcrepe \
    ffmpeg-python

# Clone RVC WebUI for training functionality
RUN git clone --depth 1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git /app/rvc-webui

# Install WebUI deps (skip problematic version-locked packages)
RUN grep -v -E "^torch==|^torchvision==|^torchaudio==|^numba==|^llvmlite==|^numpy==|^fairseq==|^faiss-cpu==|^gradio==|^fastapi==|^ffmpy==|^torchcrepe==|^pyworld==" /app/rvc-webui/requirements.txt > /tmp/rvc_reqs.txt \
    && pip install --no-cache-dir numba librosa \
    && pip install --no-cache-dir -r /tmp/rvc_reqs.txt || true \
    && pip install --no-cache-dir fairseq --no-deps || true \
    && pip install --no-cache-dir omegaconf hydra-core av pydub tensorboardX

# Install rvc package for inference
RUN pip install --no-cache-dir rvc || true

# Pre-generate matplotlib font cache
RUN python -c "import matplotlib; print('Font cache generated')" || true

# Create volume mount point
RUN mkdir -p /runpod-volume/rvc_models

# Copy handler
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]

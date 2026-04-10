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
    matplotlib==3.7.5 \
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
    && pip install --no-cache-dir omegaconf hydra-core av pydub tensorboardX \
    && pip install --no-cache-dir fairseq==0.12.2 --no-deps || true

# Fix fairseq dataclass issue with Python 3.11
# Use One-sixth's patched fork (community verified fix)
RUN pip install --no-cache-dir git+https://github.com/One-sixth/fairseq.git

# Download models using wget (more reliable than huggingface_hub in build)
RUN mkdir -p /app/rvc-webui/assets/rmvpe \
    /app/rvc-webui/assets/hubert \
    /app/rvc-webui/assets/pretrained_v2

# RMVPE (F0 extraction)
RUN wget -q -O /app/rvc-webui/assets/rmvpe/rmvpe.pt \
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt" \
    && cp /app/rvc-webui/assets/rmvpe/rmvpe.pt /app/rvc-webui/rmvpe.pt \
    && echo "RMVPE downloaded"

# HuBERT (feature extraction)
RUN wget -q -O /app/rvc-webui/assets/hubert/hubert_base.pt \
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt" \
    && echo "HuBERT downloaded"

# Pretrained v2 models (training base)
RUN wget -q -O /app/rvc-webui/assets/pretrained_v2/f0G48k.pth \
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G48k.pth" \
    && wget -q -O /app/rvc-webui/assets/pretrained_v2/f0D48k.pth \
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D48k.pth" \
    && echo "Pretrained v2 downloaded"

# Install rvc package for inference
RUN pip install --no-cache-dir rvc || true

# Pre-generate matplotlib font cache
RUN python -c "import matplotlib; print('Font cache generated')" || true

# Create volume mount point
RUN mkdir -p /runpod-volume/rvc_models

# Copy handler
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]

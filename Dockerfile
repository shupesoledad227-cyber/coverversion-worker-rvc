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
    && pip install --no-cache-dir omegaconf hydra-core av pydub tensorboardX \
    && pip install --no-cache-dir fairseq==0.12.2 --no-deps || true

# Fix fairseq dataclass issue with Python 3.11
# Use One-sixth's patched fork (community verified fix)
RUN pip install --no-cache-dir git+https://github.com/One-sixth/fairseq.git

# Download RMVPE model for F0 extraction (both locations for compatibility)
RUN mkdir -p /app/rvc-webui/assets/rmvpe && \
    python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('lj1995/VoiceConversionWebUI', 'rmvpe.pt', local_dir='/app/rvc-webui/assets/rmvpe'); \
import shutil; shutil.copy('/app/rvc-webui/assets/rmvpe/rmvpe.pt', '/app/rvc-webui/rmvpe.pt'); \
print('RMVPE downloaded')" || echo "RMVPE download skipped"

# Download HuBERT model
RUN mkdir -p /app/rvc-webui/assets/hubert && \
    python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('lj1995/VoiceConversionWebUI', 'hubert_base.pt', local_dir='/app/rvc-webui/assets/hubert'); \
print('HuBERT downloaded')" || echo "HuBERT download skipped"

# Download pretrained v2 models
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('lj1995/VoiceConversionWebUI', local_dir='/app/rvc-webui/assets', allow_patterns=['pretrained_v2/*']); \
print('Pretrained v2 downloaded')" || echo "Pretrained v2 download skipped"

# Install rvc package for inference
RUN pip install --no-cache-dir rvc || true

# Pre-generate matplotlib font cache
RUN python -c "import matplotlib; print('Font cache generated')" || true

# Create volume mount point
RUN mkdir -p /runpod-volume/rvc_models

# Copy handler
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]

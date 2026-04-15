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

# Pre-generate matplotlib font cache
RUN python -c "import matplotlib; print('Font cache generated')" || true

# Pre-download Demucs htdemucs model (~80MB)
RUN python -c "\
import torch; \
from demucs.pretrained import get_model; \
get_model('htdemucs'); \
print('htdemucs model downloaded')"

# ── MSST framework + BS Roformer + Karaoke models ──
RUN git clone --depth 1 https://github.com/ZFTurbo/Music-Source-Separation-Training.git /app/msst
RUN pip install --no-cache-dir \
    ml_collections beartype==0.14.1 rotary-embedding-torch==0.3.5 \
    einops==0.8.1 segmentation_models_pytorch==0.3.3 timm==0.9.2 \
    omegaconf wandb loralib spafe==0.3.2 auraloss torchseg \
    prodigyopt hyper_connections==0.1.11 torch_log_wmse torch_l1_snr

# BS Roformer vocals (SDR 10.87)
RUN wget -q -O /app/msst/bs_roformer_vocals.ckpt \
    "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt" \
    && wget -q -O /app/msst/bs_roformer_vocals.yaml \
    "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml" \
    && echo "BS Roformer vocals downloaded"

# Karaoke model (lead/backing separation)
RUN wget -q -O /app/msst/bs_roformer_karaoke_frazer_becruily.ckpt \
    "https://huggingface.co/becruily/bs-roformer-karaoke/resolve/main/bs_roformer_karaoke_frazer_becruily.ckpt" \
    && wget -q -O /app/msst/config_karaoke_frazer_becruily.yaml \
    "https://huggingface.co/becruily/bs-roformer-karaoke/resolve/main/config_karaoke_frazer_becruily.yaml" \
    && echo "Karaoke model downloaded"

# Create volume mount point
RUN mkdir -p /runpod-volume/rvc_models

# Copy handler
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]

# RVC v2 Train + Infer Worker for RunPod Serverless
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone RVC WebUI (contains training + inference CLI)
RUN git clone --depth 1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git /app/Retrieval-based-Voice-Conversion-WebUI

# Install RVC dependencies
RUN pip install --no-cache-dir \
    runpod \
    requests \
    pedalboard \
    demucs \
    && pip install --no-cache-dir -r /app/Retrieval-based-Voice-Conversion-WebUI/requirements.txt

# Pre-generate matplotlib font cache
RUN pip install --no-cache-dir matplotlib && python -c "import matplotlib; print('Font cache generated')" || true

# Download RVC pretrained models
RUN mkdir -p /app/Retrieval-based-Voice-Conversion-WebUI/assets/hubert && \
    python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('lj1995/VoiceConversionWebUI', 'hubert_base.pt', local_dir='/app/Retrieval-based-Voice-Conversion-WebUI/assets/hubert'); \
print('HuBERT downloaded')" || echo "HuBERT download skipped"

RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('lj1995/VoiceConversionWebUI', 'rmvpe.pt', local_dir='/app/Retrieval-based-Voice-Conversion-WebUI'); \
print('RMVPE downloaded')" || echo "RMVPE download skipped"

# Download pretrained v2 base models
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('lj1995/VoiceConversionWebUI', local_dir='/app/Retrieval-based-Voice-Conversion-WebUI/assets', allow_patterns=['pretrained_v2/*']); \
print('Pretrained v2 downloaded')" || echo "Pretrained v2 download skipped"

# Create volume mount point
RUN mkdir -p /runpod-volume/rvc_models

# Copy handler
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]

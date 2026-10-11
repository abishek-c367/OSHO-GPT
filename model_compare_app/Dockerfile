FROM python:3.11.13-slim-bookworm@sha256:cec9aa7aa96eea4fa036e9b82be1e6b325f2e3707f462d885868df51ec0a4b47

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COMPARE_DEVICE=cpu \
    COMPARE_SCRATCH_CHECKPOINT=/models/ckpt_scratch_full.pt \
    COMPARE_CONTINUED_CHECKPOINT=/models/finetuned_gpt2.pt \
    COMPARE_LORA_CHECKPOINT=/models/adapter.pt \
    COMPARE_TOKENIZER_PATH=/models/tokenizer \
    HF_HOME=/models/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

COPY model_compare_app/requirements.txt /app/model_compare_app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip==26.1.2 \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.12.0 \
    && python -m pip install --no-cache-dir -r /app/model_compare_app/requirements.txt

COPY src /app/src
COPY model_compare_app /app/model_compare_app
COPY out/ckpt_scratch_full.pt /models/ckpt_scratch_full.pt
COPY out_gpt2_osho/finetuned_gpt2.pt /models/finetuned_gpt2.pt
COPY Lora/rank_16/adapter.pt /models/adapter.pt
COPY data/Osho/gpt2 /models/tokenizer

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='gpt2', revision='607a30d783dfa663caf39e06633721c8d4cfcd7e', allow_patterns=['config.json', 'generation_config.json', 'model.safetensors'])"
RUN mkdir -p /models/huggingface/hub/models--gpt2/refs \
    && printf '%s' '607a30d783dfa663caf39e06633721c8d4cfcd7e' \
        > /models/huggingface/hub/models--gpt2/refs/main

ENV HF_HUB_OFFLINE=1

EXPOSE 8000

CMD ["uvicorn", "model_compare_app.app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "300"]

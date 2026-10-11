# Direct model comparison service

For the complete Windows-to-AWS EC2 deployment procedure, see
[AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md).

This application provides a reusable inference service and browser UI for
comparing the three trained models.

## Usage

Run from the `nanoGPT` directory:

```python
from pathlib import Path

from model_compare_app.app.model_manager import GenerationSettings, ModelManager

root = Path.cwd()
manager = ModelManager(
    scratch_checkpoint=root / "out/ckpt_scratch_full.pt",
    continued_checkpoint=root / "out_gpt2_osho/finetuned_gpt2.pt",
    lora_checkpoint=root / "Lora/rank_16/adapter.pt",
    tokenizer_path=root / "data/Osho/gpt2",
    device="cpu",
)

results = manager.generate_all(
    "What is meditation?",
    GenerationSettings(max_new_tokens=100, temperature=0.7, top_k=150),
)
for result in results:
    print(result.label, result.text)
```

The manager loads each model once and applies the same prompt and sampling
settings to all three models. A configured seed is reset before each model so
the comparison uses the same sampling sequence.

## API

Start the API from the `nanoGPT` directory:

```sh
uvicorn model_compare_app.app.api:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health` reports API status and whether the models have been loaded.
- `POST /generate` accepts `message`, `temperature`, `max_new_tokens`, `top_k`,
  and an optional `seed`, then returns all three results.
- `POST /generate/stream` sends server-sent events as each model produces each
  token, allowing the browser to display partial responses immediately.

Opening `/` serves the comparison interface. It provides:

- A prompt editor.
- Temperature, output-token, and top-k controls.
- A shared-seed toggle for controlled comparisons.
- Three responsive response cards with timing, token count, and copy actions.

Checkpoint paths and runtime settings can be overridden with
`COMPARE_SCRATCH_CHECKPOINT`, `COMPARE_CONTINUED_CHECKPOINT`,
`COMPARE_LORA_CHECKPOINT`, `COMPARE_TOKENIZER_PATH`, `COMPARE_DEVICE`, and
`COMPARE_PRETRAINED_MODEL`.

## Build and publish the Docker image

The image contains both full checkpoints, the rank-16 LoRA adapter, the
tokenizer, and the GPT-2 base weights required by that adapter. The base GPT-2
weights are downloaded from Hugging Face during the image build, so the build
needs internet access. Inference runs offline after the image is built.

The model assets alone are about 3.5 GB before compression. Allow several
gigabytes of free disk space for the build and enough container memory to load
all three models.

From the `nanoGPT` directory, build and run the image locally:

```sh
docker build -t osho-model-compare:latest -f model_compare_app/Dockerfile .
docker run --rm -p 8000:8000 --name osho-model-compare osho-model-compare:latest
```

The UI will be available at `http://localhost:8000`. To use Docker Compose
instead, run:

```sh
docker compose -f model_compare_app/docker-compose.ec2.yml up -d --build
docker compose -f model_compare_app/docker-compose.ec2.yml logs -f
```

To publish the built image to Docker Hub, replace `<dockerhub-username>` with
your account name:

```sh
docker login
docker tag osho-model-compare:latest <dockerhub-username>/osho-model-compare:latest
docker push <dockerhub-username>/osho-model-compare:latest
```

On an EC2 instance, allow inbound TCP port `8000` only from your own IP for an
initial deployment. For public use, put Nginx and HTTPS in front of the
container instead of exposing Uvicorn directly.

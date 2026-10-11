# NanoGPT: train, adapt, and deploy a GPT-2 model

This repository is a practical end-to-end GPT language-model project. It
contains the model implementation, data preparation scripts, training entry
points, a LoRA implementation, text-generation utilities, and a FastAPI web
service that compares three model variants:

1. a GPT-2-sized model trained from random initialization;
2. the same architecture continued from Hugging Face GPT-2 weights; and
3. the Hugging Face GPT-2 model adapted with low-rank adapters (LoRA).

The example dataset in this checkout is an Osho text corpus under
`data/Osho`. Replace it with text for which you have the necessary rights and
consent. Do not upload private, copyrighted, or personally identifying data to
Hugging Face, Docker Hub, or AWS without permission.

## What is in this repository?

| Path | Purpose |
| --- | --- |
| `src/model.py` | GPT architecture, Hugging Face GPT-2 weight import, generation, and LoRA layers |
| `src/train.py` | From-scratch, resume, and full-model continued training; supports single GPU and DDP |
| `src/train_lora.py` | Adapter-only LoRA fine-tuning with validation, early stopping, and resumable checkpoints |
| `utilities/generate_lora.py` | Command-line generation from a LoRA checkpoint |
| `utilities/sample.py` | Generation from a full nanoGPT checkpoint |
| `data/Osho/prepare.py` | Tokenizes a local text corpus into `train.bin` and `val.bin` |
| `data/openwebtext/prepare.py` | Downloads and tokenizes OpenWebText |
| `model_compare_app/` | FastAPI API, browser UI, Dockerfile, and EC2 deployment material |
| `config/` | Reusable training and evaluation configurations |
| `out/` | Full checkpoint output for the scratch-trained model |
| `out_gpt2_osho/` | Full checkpoint output for continued training |
| `Lora/` | LoRA adapter output |

The repository also contains notebooks and visualization utilities. They are
useful for experiments but are not required for the reproducible pipeline
below.

## 1. Prerequisites

### Hardware

Training GPT-2 is compute-intensive. A CUDA GPU with at least 12--16 GB of
VRAM is a practical starting point for GPT-2 (124M) experiments. CPU training
is useful for smoke tests only. Larger variants (`gpt2-medium`, `gpt2-large`,
and `gpt2-xl`) require substantially more VRAM or distributed training.

Inference can run on CPU, but loading all three comparison models needs
several GB of RAM. The Docker image is large because it includes full
checkpoints and a cached GPT-2 base model.

### Software

- Windows PowerShell for the local commands in this guide.
- Python 3.10 or 3.11.
- PyTorch 2.x. Install the CUDA build appropriate for the installed NVIDIA
  driver, or use the CPU wheel for inference.
- Git, and optionally Docker Desktop for local deployment.
- An AWS account and an Ubuntu EC2 instance for the final deployment.

Create an isolated environment from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy transformers datasets tiktoken tqdm
```

Install PyTorch separately so that the wheel matches your machine. For a CPU
installation:

```powershell
python -m pip install torch
```

For a CUDA installation, use the command printed by the official PyTorch
selector for your CUDA version. Verify the environment before training:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The API dependencies are pinned in
[model_compare_app/requirements.txt](model_compare_app/requirements.txt).
Install them when running the web service:

```powershell
python -m pip install -r model_compare_app\requirements.txt
```

## 2. Prepare a text dataset

The training loop expects two binary files containing GPT-2 token IDs:

```text
data\<dataset>\train.bin
data\<dataset>\val.bin
```

Each file is a `uint16` NumPy-compatible stream. The model uses GPT-2's
50,257-token vocabulary, so the same GPT-2 tokenizer must be used for
preparation, training, and inference.

### Prepare the included local corpus

`data/Osho/prepare.py` currently contains a machine-specific source path.
Change `input_file_path` to an absolute path on your machine before running
it. For example:

```powershell
notepad data\Osho\prepare.py
```

Set it to a UTF-8 text file, then run:

```powershell
python data\Osho\prepare.py
```

The script performs a 90/10 train/validation split and writes
`data/Osho/train.bin` and `data/Osho/val.bin`. It also expects a local
GPT-2 tokenizer directory. The simplest reliable option is to download it
once:

```powershell
python -c "from transformers import GPT2TokenizerFast; GPT2TokenizerFast.from_pretrained('gpt2').save_pretrained('data/Osho/gpt2')"
```

If the script's tokenizer path is not `data/Osho/gpt2`, update it before
preparing the data. Check the result:

```powershell
Get-ChildItem data\Osho\train.bin, data\Osho\val.bin, data\Osho\gpt2
```

For a real project, preserve the train/validation split across experiments,
remove duplicate documents, normalize encoding, and keep a data manifest
containing the source, license, preprocessing version, and split seed.

### Prepare OpenWebText

The OpenWebText script downloads a large dataset and can require tens of GB
of cache and disk space. It uses `datasets`, `tiktoken`, multiple processes,
and writes `data/openwebtext/train.bin` and `data/openwebtext/val.bin`:

```powershell
python data\openwebtext\prepare.py
```

Do not run this command for a small smoke test. Start with the local corpus or
a small, licensed sample.

## 3. Train a GPT model from scratch

`src/train.py` is the main nanoGPT training loop. From the repository root,
run a short smoke test first:

```powershell
python src\train.py `
  --dataset=Osho `
  --out_dir=out\scratch_smoke `
  --init_from=scratch `
  --batch_size=2 `
  --block_size=256 `
  --gradient_accumulation_steps=1 `
  --max_iters=20 `
  --eval_interval=10 `
  --eval_iters=5 `
  --learning_rate=6e-4 `
  --device=cuda `
  --compile=False
```

Use `--device=cpu` for a CPU smoke test. The script reads configuration
overrides through `utilities/configurator.py`; the command-line values above
take precedence over the defaults.

For a longer GPT-2-sized run, make a copy of the configuration and tune it
for the available GPU:

```powershell
Copy-Item config\train_gpt2.py config\train_osho_scratch.py
notepad config\train_osho_scratch.py
Copy-Item utilities\configurator.py .\configurator.py
python src\train.py config\train_osho_scratch.py --init_from=scratch --dataset=Osho --out_dir=out\scratch
```

The `Copy-Item` command supplies the compatibility shim expected by the
legacy `exec(open('configurator.py'))` line in `src/train.py`. Keep that file
at the repository root while using this entry point. The configuration file is
passed as a positional argument; individual values use `--key=value`.
The important scratch settings are:

```text
init_from = "scratch"
dataset = "Osho"
out_dir = "out/scratch"
n_layer = 12
n_head = 12
n_embd = 768
block_size = 1024
```

For multi-GPU training on one Windows or Linux host, launch the same script
through `torchrun`:

```powershell
torchrun --standalone --nproc_per_node=2 src\train.py `
  --dataset=Osho `
  --out_dir=out\scratch `
  --init_from=scratch `
  --batch_size=2 `
  --gradient_accumulation_steps=8 `
  --max_iters=10000
```

The effective token batch is:

```text
batch_size * gradient_accumulation_steps * number_of_GPUs * block_size
```

Increase it gradually until the GPU is well utilized without out-of-memory
errors. The training loop periodically evaluates `train` and `val` loss and
saves `ckpt_scratch_full.pt` when validation loss improves. A checkpoint
contains the model weights, model configuration, optimizer state, iteration,
and validation metric, so it can be resumed.

Resume a run without changing the architecture:

```powershell
python src\train.py `
  --dataset=Osho `
  --out_dir=out\scratch `
  --init_from=resume `
  --max_iters=20000
```

Do not overwrite a good checkpoint while experimenting; use a new `out_dir`.

## 4. Continue training from Hugging Face GPT-2

The `GPT.from_pretrained` method in `src/model.py` downloads a Hugging Face
GPT-2 checkpoint and converts its Conv1D weights into this nanoGPT model's
Linear layout. Supported identifiers are `gpt2`, `gpt2-medium`, `gpt2-large`,
and `gpt2-xl`. The first run needs internet access; later runs use the local
Hugging Face cache.

Continued training updates all model parameters. Use a smaller learning rate
than scratch training and monitor validation loss for overfitting:

```powershell
python src\train.py `
  --dataset=Osho `
  --out_dir=out_gpt2_osho `
  --init_from=gpt2 `
  --batch_size=2 `
  --block_size=512 `
  --gradient_accumulation_steps=16 `
  --learning_rate=3e-5 `
  --max_iters=2000 `
  --eval_interval=100 `
  --eval_iters=20 `
  --decay_lr=False `
  --device=cuda
```

For `gpt2-medium`, `gpt2-large`, or `gpt2-xl`, replace `--init_from` and
reduce the batch size or use DDP. The output is still a full nanoGPT
checkpoint and can be sampled with the regular generation utility.

Important compatibility rules:

- Keep the GPT-2 tokenizer and vocabulary at 50,257 tokens.
- Do not change `n_layer`, `n_head`, or `n_embd` when loading a named GPT-2
  checkpoint.
- `block_size` may be cropped below 1,024, but a checkpoint cropped to a
  smaller value cannot accept longer sequences.
- Use the same checkpoint and tokenizer pair during deployment.

## 5. LoRA fine-tuning of the same GPT-2 model

LoRA freezes the Hugging Face GPT-2 base weights and trains small low-rank
updates around the attention and MLP linear layers. It uses far less optimizer
memory than full continued training and produces a small adapter checkpoint.
The base model is still required at inference time.

Run a short LoRA smoke test:

```powershell
python src\train_lora.py `
  --dataset=Osho `
  --out_dir=Lora\smoke `
  --init_from=gpt2 `
  --batch_size=2 `
  --gradient_accumulation_steps=4 `
  --block_size=512 `
  --max_iters=20 `
  --eval_interval=10 `
  --eval_iters=5 `
  --checkpoint_interval=10 `
  --early_stopping_patience=0 `
  --lora_rank=8 `
  --lora_alpha=16 `
  --lora_dropout=0.05 `
  --device=cuda
```

Run a normal adapter training job:

```powershell
python src\train_lora.py `
  --dataset=Osho `
  --out_dir=Lora\rank_16 `
  --init_from=gpt2 `
  --batch_size=2 `
  --gradient_accumulation_steps=16 `
  --block_size=512 `
  --max_iters=2000 `
  --eval_interval=100 `
  --eval_iters=20 `
  --checkpoint_interval=100 `
  --early_stopping_patience=3 `
  --learning_rate=2e-4 `
  --lora_rank=16 `
  --lora_alpha=32 `
  --lora_dropout=0.05 `
  --device=cuda
```

The best adapter is written to `Lora\rank_16\adapter.pt`; `config.json`
records the run arguments. Periodic checkpoints include enough metadata to
resume:

```powershell
python src\train_lora.py `
  --dataset=Osho `
  --out_dir=Lora\rank_16 `
  --init_from=gpt2 `
  --resume_from=Lora\rank_16\adapter.pt `
  --max_iters=4000 `
  --device=cuda
```

The resume arguments must agree with the checkpoint's model name, block size,
rank, alpha, and dropout. For a fair comparison with full continued training,
use the same tokenized dataset, validation split, tokenizer, and prompt set.

Generate from the adapter:

```powershell
python utilities\generate_lora.py `
  --message "What is meditation?" `
  --checkpoint=Lora\rank_16\adapter.pt `
  --tokenizer=data\Osho\gpt2 `
  --model=gpt2 `
  --max-new-tokens=100 `
  --temperature=0.7 `
  --top-k=150 `
  --device=cuda
```

## 6. Evaluate and compare models locally

The comparison application loads the scratch checkpoint, full continued
checkpoint, and LoRA adapter with shared generation settings:

```powershell
uvicorn model_compare_app.app.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The API can also be called directly:

```powershell
$body = @{
  message = "What is meditation?"
  max_new_tokens = 50
  temperature = 0.7
  top_k = 150
  seed = 1337
} | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/generate `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

`GET /health` reports whether models have been loaded. Model loading is lazy,
so the first `/generate` request can be slow. `POST /generate/stream` emits
server-sent events while each model generates.

## 7. Build and test the Docker image

The Dockerfile copies the three trained artifacts and tokenizer into the
image, downloads a pinned GPT-2 base snapshot during the build, and enables
offline Hugging Face mode at runtime. Run from the repository root:

```powershell
docker build -t osho-model-compare:latest -f model_compare_app\Dockerfile .
docker run --rm -p 8000:8000 --name osho-model-compare osho-model-compare:latest
```

The build requires these files to exist:

```text
out/ckpt_scratch_full.pt
out_gpt2_osho/finetuned_gpt2.pt
Lora/rank_16/adapter.pt
data/Osho/gpt2/
```

If your output directories differ, either copy the chosen artifacts to those
paths or update the `COPY` instructions and environment variables in
[model_compare_app/Dockerfile](model_compare_app/Dockerfile).

Use Compose for repeatable local runs:

```powershell
docker compose -f model_compare_app\docker-compose.ec2.yml up -d --build
docker compose -f model_compare_app\docker-compose.ec2.yml logs -f
```

The image contains model weights and should be treated as sensitive
deployment data. Do not publish it to a public registry unless the model and
training data permit that distribution.

## 8. Deploy to AWS EC2

The detailed Windows-to-Ubuntu procedure is in
[model_compare_app/AWS_DEPLOYMENT.md](model_compare_app/AWS_DEPLOYMENT.md).
The following is the complete high-level sequence.

### Provision and secure the instance

1. Create an Ubuntu EC2 instance with enough RAM and disk for the image.
   CPU inference is the default; use a GPU instance only when latency
   justifies its cost.
2. Attach a security group allowing TCP 22 only from your workstation IP.
3. Allow TCP 8000 only from your own IP during testing. For public production,
   put a reverse proxy and HTTPS in front of the application instead of
   exposing Uvicorn directly.
4. Install Docker and the Compose plugin on Ubuntu.
5. Store the `.pem` key outside the repository or ensure it is ignored and
   never committed.

### Upload only the deployment inputs

From Windows PowerShell, replace the host and key path:

```powershell
$key = "C:\Keys\nanogpt-ec2.pem"
$hostName = "ubuntu@203.0.113.10"

ssh -i $key $hostName "mkdir -p /home/ubuntu/nanoGPT"
scp -r -i $key `
  .\model_compare_app `
  .\src `
  .\out `
  .\out_gpt2_osho `
  .\Lora `
  .\data\Osho `
  "${hostName}:/home/ubuntu/nanoGPT/"
```

Verify the expected paths on the instance:

```powershell
ssh -i $key $hostName `
  "cd /home/ubuntu/nanoGPT && ls -lh out/ckpt_scratch_full.pt out_gpt2_osho/finetuned_gpt2.pt Lora/rank_16/adapter.pt data/Osho/gpt2/config.json"
```

### Build, start, and verify

Run on the EC2 host:

```bash
cd /home/ubuntu/nanoGPT
docker compose -f model_compare_app/docker-compose.ec2.yml build
docker compose -f model_compare_app/docker-compose.ec2.yml up -d
docker compose -f model_compare_app/docker-compose.ec2.yml ps
curl -i http://127.0.0.1:8000/health
```

Test generation from PowerShell:

```powershell
$body = @{ message = "Hello"; max_new_tokens = 1; temperature = 0.7; top_k = 10; seed = 1 } |
  ConvertTo-Json -Compress
Invoke-RestMethod `
  -Uri "http://203.0.113.10:8000/generate" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body `
  -TimeoutSec 180
```

Expected results contain `scratch`, `continued_training`, and `lora_rank_16`.
The first request loads all models and can take several minutes.

Useful operations:

```bash
docker compose -f model_compare_app/docker-compose.ec2.yml logs -f
docker compose -f model_compare_app/docker-compose.ec2.yml restart
docker compose -f model_compare_app/docker-compose.ec2.yml down
```

### Production checklist

- Use an IAM role with the minimum permissions; do not put AWS access keys in
  the source tree or Docker image.
- Keep port 22 restricted and use a fixed administrator IP or a bastion.
- Put Nginx or an AWS load balancer with TLS in front of Uvicorn.
- Add authentication and request rate limiting before making `/generate`
  public.
- Set CloudWatch logging, disk alerts, and an instance stop schedule.
- Pin image versions and model revisions, and retain a tested rollback image.
- Load-test with bounded `max_new_tokens`; generation cost scales with output
  length and concurrent requests.

## 9. Troubleshooting

### `FileNotFoundError` for `train.bin` or `val.bin`

Run the matching dataset preparation script and pass the same dataset name to
the trainer:

```powershell
python data\Osho\prepare.py
python src\train_lora.py --dataset=Osho --max_iters=1 --device=cpu
```

### CUDA out-of-memory

Reduce `batch_size`, `block_size`, or
`gradient_accumulation_steps` (the latter reduces the effective batch size),
choose a smaller GPT-2 variant, and close other GPU processes. Gradient
accumulation can be increased again after the per-step batch fits.

### Hugging Face download or offline errors

The first `--init_from=gpt2` run must have network access. For deployment,
the Dockerfile downloads the pinned snapshot during `docker build`; the
runtime then uses `HF_HUB_OFFLINE=1`. Do not enable offline mode before the
base snapshot exists locally.

### Checkpoint/tokenizer mismatch

Use the tokenizer generated from the same GPT-2 vocabulary and do not mix a
custom tokenizer with GPT-2 weights. LoRA adapters are not standalone models:
they require the matching base model name and LoRA metadata.

### API returns a model-loading error

Inspect the container logs and verify all `COPY` source files exist on the
build machine:

```bash
docker logs --tail 200 osho-model-compare
```

## 10. Reproducibility and experiment records

For every run, record:

- Git commit and configuration file.
- Dataset source, license, preprocessing version, and split seed.
- Tokenizer and Hugging Face model revision.
- GPU type, precision, batch size, sequence length, optimizer, learning rate,
  LoRA rank/alpha/dropout, and number of tokens.
- Training and validation loss curves.
- Evaluation prompts and generation settings.

Keep large checkpoints in object storage or a model registry rather than Git.
This repository's ignore rules intentionally exclude local environments,
caches, logs, generated binaries, model checkpoints, and private keys.

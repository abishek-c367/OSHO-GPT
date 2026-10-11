# AWS EC2 Deployment Guide

This guide deploys the NanoGPT model comparison website to an Ubuntu EC2
instance from a Windows PowerShell workstation. It covers the FastAPI gateway,
browser UI, two full checkpoints, the LoRA adapter, tokenizer files, and the
offline GPT-2 base model used by the LoRA model.

## What Was Deployed

The application is served by the `model_compare_app` FastAPI gateway. The
Docker image contains:

- `model_compare_app`: FastAPI routes and browser UI.
- `src`: the NanoGPT model implementation.
- `out/ckpt_scratch_full.pt`: scratch model checkpoint.
- `out_gpt2_osho/finetuned_gpt2.pt`: continued-training checkpoint.
- `Lora/rank_16/adapter.pt`: LoRA adapter checkpoint.
- `data/Osho/gpt2`: tokenizer files.
- A cached GPT-2 base model downloaded during the image build.

The website is exposed on port `8000`.

## Requirements

Local workstation:

- Windows PowerShell.
- OpenSSH client (`ssh` and `scp`).
- The EC2 private key file. Never commit or publish this file.

EC2 instance:

- Ubuntu, accessed as `ubuntu`.
- Docker and Docker Compose.
- At least 8 GB RAM is recommended for loading all three models.
- At least 10 GB free disk space after the project files are uploaded. The
  checkpoints and Docker layers require several gigabytes.

## 1. Configure EC2 Security Groups

In the AWS Console:

1. Open **EC2**, select the correct Region, and select the instance.
2. Open the instance **Security** tab.
3. Open the security group linked to that instance.
4. Edit **Inbound rules**.
5. Add these rules:

| Type | Port | Source | Purpose |
| --- | ---: | --- | --- |
| SSH | 22 | My IP, or your fixed `/32` address | SSH and file upload |
| Custom TCP | 8000 | My IP for private use, or `0.0.0.0/0` for public use | Website and API |

For EC2 Instance Connect in the browser, AWS may also require the Region's
EC2 Instance Connect service CIDR for port 22. Use the CIDR shown by the AWS
warning for the selected Region. This is separate from SSH access using the
private key.

After saving, verify from PowerShell:

```powershell
Test-NetConnection 13.127.96.164 -Port 22
Test-NetConnection 13.127.96.164 -Port 8000
```

Both commands should show `TcpTestSucceeded : True`.

## 2. Prepare the Private Key on Windows

Windows OpenSSH rejects a private key that is readable by other users. From
the project root, restrict the key ACL:

```powershell
$key = (Resolve-Path '.\Private_key_for_aws_oshogpt.pem').Path
icacls $key /inheritance:r /grant:r "${env:USERNAME}:(R)"
```

Do not print the key contents and do not put the key in the Docker build
context.

Test SSH access:

```powershell
ssh -o ConnectTimeout=10 -i $key ubuntu@13.127.96.164 "hostname; whoami; docker --version; docker compose version"
```

Expected output includes the `ubuntu` user and Docker versions.

## 3. Upload the Required Files

Run from the `nanoGPT` project root. This uploads only the directories needed
by the Dockerfile instead of notebooks, logs, the local virtual environment,
and unrelated training artifacts.

```powershell
ssh -i $key ubuntu@13.127.96.164 "mkdir -p /home/ubuntu/nanoGPT"

scp -r -i $key `
  '.\model_compare_app' `
  '.\src' `
  '.\out' `
  '.\out_gpt2_osho' `
  '.\Lora' `
  '.\data\Osho' `
  ubuntu@13.127.96.164:/home/ubuntu/nanoGPT/
```

`scp` places each source directory by its final directory name. If the
`data\Osho` source lands at `/home/ubuntu/nanoGPT/Osho`, correct it before
building:

```powershell
ssh -i $key ubuntu@13.127.96.164 `
  "cd /home/ubuntu/nanoGPT; mkdir -p data; mv Osho data/Osho"
```

Verify the important files remotely:

```powershell
ssh -i $key ubuntu@13.127.96.164 `
  "cd /home/ubuntu/nanoGPT; ls -lh out/ckpt_scratch_full.pt out_gpt2_osho/finetuned_gpt2.pt Lora/rank_16/adapter.pt data/Osho/gpt2/config.json"
```

## 4. Build the Docker Image

The Dockerfile downloads the GPT-2 base files during the build and sets
`HF_HUB_OFFLINE=1` in the final image. The build machine therefore needs
internet access, but inference does not.

From the remote project directory:

```bash
cd /home/ubuntu/nanoGPT
docker compose -f model_compare_app/docker-compose.ec2.yml build
```

The first build can take several minutes because it transfers approximately
3 GB of model files and installs PyTorch and Transformers. Later builds reuse
Docker's cached layers.

If Docker is waiting while resolving the pinned base image, confirm that the
instance can reach Docker Hub. Do not delete the existing image while
debugging; the existing image can keep the service available.

## 5. Start the Service

```bash
cd /home/ubuntu/nanoGPT
docker compose -f model_compare_app/docker-compose.ec2.yml up -d
docker compose -f model_compare_app/docker-compose.ec2.yml ps
```

The container should be named `osho-model-compare` and show a port mapping
similar to `0.0.0.0:8000->8000/tcp`.

## 6. Verify the Deployment

Verify from the EC2 instance first:

```bash
curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/
```

Verify from the workstation:

```powershell
Invoke-WebRequest 'http://13.127.96.164:8000/health' -UseBasicParsing
Invoke-WebRequest 'http://13.127.96.164:8000/' -UseBasicParsing
```

The health response before the first generation request is normally:

```json
{"status":"ok","models_loaded":false}
```

Model loading is lazy. Test generation with a small request first:

```powershell
$body = @{
  message = 'Hello'
  max_new_tokens = 1
  temperature = 0.7
  top_k = 10
  seed = 1
} | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Uri 'http://13.127.96.164:8000/generate' `
  -Method Post `
  -ContentType 'application/json' `
  -Body $body `
  -TimeoutSec 180
```

The response should contain three entries with `generated_tokens: 1`:

- `scratch`
- `continued_training`
- `lora_rank_16`

The browser UI is available at:

```text
http://13.127.96.164:8000
```

## 7. Useful Operations

View application logs:

```bash
docker logs --tail 200 -f osho-model-compare
```

Restart after a configuration change:

```bash
cd /home/ubuntu/nanoGPT
docker compose -f model_compare_app/docker-compose.ec2.yml up -d --force-recreate --no-build
```

Check container memory and restart state:

```bash
docker stats --no-stream osho-model-compare
docker inspect -f 'status={{.State.Status}} oom={{.State.OOMKilled}} restarts={{.RestartCount}}' osho-model-compare
```

Stop the service:

```bash
docker compose -f model_compare_app/docker-compose.ec2.yml down
```

## Problems Encountered and Solutions

### The bundled virtual environment was stale

The local `env\pyvenv.cfg` pointed to a removed Conda interpreter. Its Python
executable could not start. Python 3.11 was already installed system-wide and
already had the matching CPU PyTorch version, so the gateway dependencies were
installed into that interpreter instead. This did not affect the Docker
deployment, which installs its own dependencies from `requirements.txt`.

### OpenSSH rejected the private key

Windows reported `UNPROTECTED PRIVATE KEY FILE` because the key ACL allowed
other users to read it. Removing inherited permissions and granting only the
current Windows user read access fixed SSH authentication:

```powershell
icacls $key /inheritance:r /grant:r "${env:USERNAME}:(R)"
```

### The uploaded Osho directory was one level too high

Uploading several directories in one `scp` command placed `Osho` directly in
the destination. The Dockerfile expects `data/Osho`, so the remote directory
was moved into `data` before building.

### The website was unreachable even though the container was running

The service listened on `0.0.0.0:8000` inside EC2, but the AWS security group
did not initially allow inbound TCP `8000`. Adding the Custom TCP rule fixed
website access. Port 22 also had to be allowed from the workstation's current
public IP before SSH diagnostics could be run again.

### EC2 Instance Connect showed a port 22 warning

That warning refers to the browser-based EC2 Instance Connect service, not the
`.pem` SSH connection. For normal key-based SSH, allow port 22 from `My IP`.
For EC2 Instance Connect, add the Region-specific service CIDR shown by AWS.

### Generation returned HTTP 500 while the UI loaded correctly

The UI and `/health` endpoint do not load model weights. The first generation
request loads all three models. Logs showed that the LoRA model called
`GPT2LMHeadModel.from_pretrained("gpt2")` while the image was offline. The
GPT-2 files existed in the image, but the repository-name lookup could not
resolve the cached snapshot.

The durable fix has two parts:

1. `src/model.py` accepts a local GPT-2 snapshot directory as the pretrained
   model source.
2. `model_compare_app/docker-compose.ec2.yml` sets
   `COMPARE_PRETRAINED_MODEL` to the cached GPT-2 snapshot path.

The Dockerfile also creates the Hugging Face `refs/main` pointer for clean
offline builds. This prevents runtime network access from being required.

### A first attempted cache-pointer fix was not enough

Adding only `refs/main` did not fix Transformers' repository-name resolution in
the running image. The model loader was therefore changed to use the explicit
local snapshot path. This is why the source-code fix is preferred over a
container-only patch or manual `docker commit`.

### Generation was initially suspected to be an out-of-memory failure

The instance had limited free memory, so OOM was checked with `docker inspect`,
`docker stats`, and host memory information. The container reported
`OOMKilled=false`, and the logs identified the offline Hugging Face lookup as
the actual cause.

## Security and Reliability Notes

- Prefer `My IP` or a fixed `/32` source instead of `0.0.0.0/0` for SSH.
- Restrict port `8000` to trusted IPs unless public access is required.
- Do not commit the `.pem` file or include it in Docker images.
- The public EC2 address can change after a stop/start. Use an Elastic IP for
  a stable URL.
- For production, put Nginx or an AWS load balancer with HTTPS in front of
  Uvicorn instead of exposing port 8000 directly.
- The Compose service uses `restart: unless-stopped`, so it restarts after a
  Docker daemon or instance restart. It does not protect against a changed
  public IP or a deleted security-group rule.
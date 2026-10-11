"""Generate text from a GPT-2 LoRA checkpoint."""

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import GPT2TokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import GPT, apply_lora


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message", required=True, help="User message used as the prompt")
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "Lora" / "rank_16" / "adapter.pt"),
        help="LoRA adapter or resumable checkpoint",
    )
    parser.add_argument("--tokenizer", default=str(ROOT / "data" / "Osho" / "gpt2"))
    parser.add_argument("--model", default="gpt2", choices=("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"))
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    rank = int(checkpoint.get("lora_rank", 16))
    alpha = float(checkpoint.get("lora_alpha", 32.0))
    dropout = float(checkpoint.get("lora_dropout", 0.05))
    block_size = int(checkpoint.get("block_size", 1024))

    model = GPT.from_pretrained(args.model, {"dropout": 0.0})
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
    model = apply_lora(model, rank=rank, alpha=alpha, dropout=dropout)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval().to(device)

    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer, local_files_only=True)
    prompt = f"{args.message}"
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) > model.config.block_size:
        token_ids = token_ids[-model.config.block_size:]
    context = torch.tensor(token_ids, dtype=torch.long, device=device)[None, :]

    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
        if device.type == "cuda"
        else torch.float32
    )
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), autocast_context:
        generated = model.generate(
            context,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    print(tokenizer.decode(generated[0].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()

"""Single-file LoRA fine-tuning entry point for the GPT-2 nanoGPT model.

Example:
    python src/train_lora.py --max_iters=1000

The base GPT-2 weights are frozen. Only LoRA adapter weights are optimized and
saved in ``Lora/adapter.pt``.
"""

import argparse
import json
import math
import pickle
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, apply_lora, lora_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tune GPT-2 on tokenized binary data")
    parser.add_argument("--dataset", default="Osho")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--out_dir", default=str(ROOT / "Lora"))
    parser.add_argument("--resume_from", default=None,
                        help="Load an adapter or full checkpoint and continue to max_iters")
    parser.add_argument("--init_from", default="gpt2",
                        choices=("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--checkpoint_interval", type=int, default=100,
                        help="Save a resumable checkpoint every N iterations")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Stop after this many evaluations without validation improvement; 0 disables")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0,
                        help="Minimum validation-loss improvement required to reset patience")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir or ROOT / "data" / args.dataset)
    out_dir = Path(args.out_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = data_dir / "meta.pkl"
    if meta_path.exists():
        with meta_path.open("rb") as handle:
            vocab_size = pickle.load(handle)["vocab_size"]
        if vocab_size != 50257:
            raise ValueError("The GPT-2 tokenizer requires a 50257-token dataset")
    train_path, val_path = data_dir / "train.bin", data_dir / "val.bin"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Expected train.bin and val.bin in {data_dir}")

    torch.manual_seed(1337)
    model = GPT.from_pretrained(args.init_from, {"dropout": 0.0})
    if args.block_size < model.config.block_size:
        model.crop_block_size(args.block_size)
    model = apply_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable LoRA parameters: {sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
    resume_step = 0
    best_val = math.inf
    evaluations_without_improvement = 0
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume = torch.load(resume_path, map_location="cpu")
        expected_metadata = {
            "model_name": args.init_from,
            "block_size": args.block_size,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        }
        for key, expected in expected_metadata.items():
            if key in resume and resume[key] != expected:
                raise ValueError(
                    f"{resume_path} was created with {key}={resume[key]!r}, "
                    f"but this run uses {expected!r}"
                )
        model.load_state_dict(resume["model"], strict=False)
        resume_step = int(resume.get("step", 0))
        best_val = float(resume.get("best_val", resume.get("losses", {}).get("val", math.inf)))
        print(f"resumed LoRA weights from {resume_path} at step {resume_step}")

    def get_batch(split):
        path = train_path if split == "train" else val_path
        data = np.memmap(path, dtype=np.uint16, mode="r")
        if len(data) <= args.block_size:
            raise ValueError(f"{path} must contain more than block_size tokens")
        starts = torch.randint(len(data) - args.block_size, (args.batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + args.block_size].astype(np.int64))
                         for i in starts])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + args.block_size].astype(np.int64))
                         for i in starts])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = {}
        for split in ("train", "val"):
            values = []
            for _ in range(args.eval_iters):
                x, y = get_batch(split)
                with ctx:
                    _, loss = model(x, y)
                values.append(loss.item())
            losses[split] = sum(values) / len(values)
        model.train()
        return losses

    def save_checkpoint(step, losses, path, include_optimizer=False):
        checkpoint = {
            "model": lora_state_dict(model),
            "model_name": args.init_from,
            "block_size": args.block_size,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "step": step,
            "losses": losses,
            "best_val": best_val,
        }
        if include_optimizer:
            checkpoint["optimizer"] = optimizer.state_dict()
            checkpoint["scaler"] = scaler.state_dict()
        torch.save(checkpoint, path)
        (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
        print(f"saved checkpoint to {path}")

    if args.resume_from:
        resume = torch.load(args.resume_from, map_location="cpu")
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])
        if "scaler" in resume:
            scaler.load_state_dict(resume["scaler"])
    x, y = get_batch("train")
    start = time.time()
    for step in range(resume_step, args.max_iters + 1):
        if step % args.eval_interval == 0:
            losses = evaluate()
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses["val"] < best_val - args.early_stopping_min_delta:
                best_val = losses["val"]
                evaluations_without_improvement = 0
                save_checkpoint(step, losses, out_dir / "adapter.pt")
            else:
                evaluations_without_improvement += 1
                print(f"validation has not improved for {evaluations_without_improvement} "
                      f"evaluation(s)")
            if (args.early_stopping_patience > 0
                    and evaluations_without_improvement >= args.early_stopping_patience):
                print("early stopping: validation loss stopped improving")
                break
        if step == args.max_iters:
            break
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.gradient_accumulation_steps):
            with ctx:
                _, loss = model(x, y)
                loss = loss / args.gradient_accumulation_steps
            x, y = get_batch("train")
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if args.checkpoint_interval > 0 and (step + 1) % args.checkpoint_interval == 0:
            save_checkpoint(
                step + 1,
                {"train": loss.item() * args.gradient_accumulation_steps},
                out_dir / f"checkpoint_step_{step + 1}.pt",
                include_optimizer=True,
            )
        if step % 10 == 0:
            print(f"iter {step}: loss {loss.item() * args.gradient_accumulation_steps:.4f}, "
                  f"elapsed {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()

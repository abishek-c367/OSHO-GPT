"""Load and run the three trained nanoGPT models with shared settings."""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from transformers import GPT2TokenizerFast


NANO_GPT_ROOT = Path(__file__).resolve().parents[2]
if str(NANO_GPT_ROOT) not in sys.path:
    sys.path.insert(0, str(NANO_GPT_ROOT))

from src.model import GPT, GPTConfig, apply_lora


@dataclass(frozen=True)
class GenerationSettings:
    """Sampling controls shared by every model in one comparison."""

    max_new_tokens: int = 100
    temperature: float = 0.7
    top_k: int = 150
    seed: int | None = 1337


@dataclass(frozen=True)
class ModelResult:
    """One model's continuation and basic generation metadata."""

    model: str
    label: str
    text: str
    generated_tokens: int
    duration_ms: float


@dataclass(frozen=True)
class GenerationEvent:
    """One progress event emitted while a model is generating."""

    type: str
    model: str
    label: str
    text: str = ""
    generated_tokens: int = 0
    duration_ms: float = 0.0


@dataclass(frozen=True)
class _ModelSpec:
    name: str
    label: str
    model: GPT


class ModelManager:
    """Own all three models and generate comparable continuations."""

    def __init__(
        self,
        scratch_checkpoint: Path,
        continued_checkpoint: Path,
        lora_checkpoint: Path,
        tokenizer_path: Path,
        device: str = "cpu",
        pretrained_model: str = "gpt2",
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available")

        self.tokenizer = self._load_tokenizer(tokenizer_path)
        self.models = (
            _ModelSpec(
                "scratch",
                "Scratch",
                self._load_full_checkpoint(scratch_checkpoint),
            ),
            _ModelSpec(
                "continued_training",
                "Continued Training",
                self._load_full_checkpoint(continued_checkpoint),
            ),
            _ModelSpec(
                "lora_rank_16",
                "LoRA Fine-tuning",
                self._load_lora_checkpoint(lora_checkpoint, pretrained_model),
            ),
        )

    def generate_all(
        self,
        prompt: str,
        settings: GenerationSettings | None = None,
    ) -> list[ModelResult]:
        """Generate one continuation from each model using identical controls."""

        settings = settings or GenerationSettings()
        self._validate_request(prompt, settings)
        results: list[ModelResult] = []

        for spec in self.models:
            if settings.seed is not None:
                torch.manual_seed(settings.seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(settings.seed)

            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            available = spec.model.config.block_size - settings.max_new_tokens
            prompt_ids = prompt_ids[-available:]
            input_ids = torch.tensor(
                prompt_ids, dtype=torch.long, device=self.device
            )[None, :]

            started = time.perf_counter()
            with torch.inference_mode():
                output = spec.model.generate(
                    input_ids,
                    max_new_tokens=settings.max_new_tokens,
                    temperature=settings.temperature,
                    top_k=settings.top_k,
                )
            duration_ms = (time.perf_counter() - started) * 1000
            generated_ids = output[0, len(prompt_ids) :].tolist()
            results.append(
                ModelResult(
                    model=spec.name,
                    label=spec.label,
                    text=self.tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    ).strip(),
                    generated_tokens=len(generated_ids),
                    duration_ms=round(duration_ms, 2),
                )
            )

        return results

    def generate_stream(
        self,
        prompt: str,
        settings: GenerationSettings | None = None,
    ) -> Iterator[GenerationEvent]:
        """Yield model lifecycle and partial-text events during generation."""

        settings = settings or GenerationSettings()
        self._validate_request(prompt, settings)

        for spec in self.models:
            if settings.seed is not None:
                torch.manual_seed(settings.seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(settings.seed)

            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            available = spec.model.config.block_size - settings.max_new_tokens
            prompt_ids = prompt_ids[-available:]
            input_ids = torch.tensor(
                prompt_ids, dtype=torch.long, device=self.device
            )[None, :]
            generated_ids: list[int] = []
            started = time.perf_counter()
            yield GenerationEvent("start", spec.name, spec.label)

            with torch.inference_mode():
                for _ in range(settings.max_new_tokens):
                    idx_cond = (
                        input_ids
                        if input_ids.size(1) <= spec.model.config.block_size
                        else input_ids[:, -spec.model.config.block_size :]
                    )
                    logits, _ = spec.model(idx_cond)
                    logits = logits[:, -1, :] / settings.temperature
                    v, _ = torch.topk(
                        logits, min(settings.top_k, logits.size(-1))
                    )
                    logits[logits < v[:, [-1]]] = -float("Inf")
                    next_id = torch.multinomial(
                        torch.nn.functional.softmax(logits, dim=-1),
                        num_samples=1,
                    )
                    input_ids = torch.cat((input_ids, next_id), dim=1)
                    generated_ids.append(int(next_id.item()))
                    yield GenerationEvent(
                        "token",
                        spec.name,
                        spec.label,
                        text=self.tokenizer.decode(
                            generated_ids, skip_special_tokens=True
                        ),
                        generated_tokens=len(generated_ids),
                    )

            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            yield GenerationEvent(
                "done",
                spec.name,
                spec.label,
                text=self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                ).strip(),
                generated_tokens=len(generated_ids),
                duration_ms=duration_ms,
            )

    def _load_tokenizer(self, tokenizer_path: Path) -> GPT2TokenizerFast:
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_path}")
        return GPT2TokenizerFast.from_pretrained(
            str(tokenizer_path), local_files_only=True
        )

    def _load_full_checkpoint(self, checkpoint_path: Path) -> GPT:
        checkpoint = _load_checkpoint(checkpoint_path, self.device)
        try:
            model = GPT(GPTConfig(**checkpoint["model_args"]))
            state_dict = checkpoint["model"]
        except KeyError as exc:
            raise ValueError(
                f"Full checkpoint is missing required field: {exc.args[0]}"
            ) from exc
        state_dict = {
            key.removeprefix("_orig_mod."): value
            for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict)
        return model.eval().to(self.device)

    def _load_lora_checkpoint(
        self, checkpoint_path: Path, pretrained_model: str
    ) -> GPT:
        checkpoint = _load_checkpoint(checkpoint_path, torch.device("cpu"))
        try:
            rank = int(checkpoint["lora_rank"])
            alpha = float(checkpoint["lora_alpha"])
            dropout = float(checkpoint["lora_dropout"])
            adapter_state = checkpoint["model"]
        except KeyError as exc:
            raise ValueError(
                f"LoRA checkpoint is missing required field: {exc.args[0]}"
            ) from exc

        model = GPT.from_pretrained(pretrained_model, {"dropout": 0.0})
        model = apply_lora(model, rank=rank, alpha=alpha, dropout=dropout)
        model.load_state_dict(adapter_state, strict=False)
        return model.eval().to(self.device)

    def _validate_request(
        self, prompt: str, settings: GenerationSettings
    ) -> None:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if settings.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if settings.temperature <= 0 or not math.isfinite(settings.temperature):
            raise ValueError("temperature must be a finite value greater than zero")
        if settings.top_k < 1:
            raise ValueError("top_k must be positive")
        for spec in self.models:
            if settings.max_new_tokens >= spec.model.config.block_size:
                raise ValueError(
                    "max_new_tokens must leave room for at least one prompt token"
                )
            if settings.top_k > spec.model.config.vocab_size:
                raise ValueError(
                    f"top_k cannot exceed {spec.model.config.vocab_size}"
                )


def _load_checkpoint(path: Path, map_location: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

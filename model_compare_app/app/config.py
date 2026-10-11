"""Environment-backed configuration for the comparison API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
NANO_GPT_ROOT = APP_ROOT.parent


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    path = Path(value) if value else default
    return path if path.is_absolute() else (NANO_GPT_ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    scratch_checkpoint: Path = NANO_GPT_ROOT / "out" / "ckpt_scratch_full.pt"
    continued_checkpoint: Path = (
        NANO_GPT_ROOT / "out_gpt2_osho" / "finetuned_gpt2.pt"
    )
    lora_checkpoint: Path = NANO_GPT_ROOT / "Lora" / "rank_16" / "adapter.pt"
    tokenizer_path: Path = NANO_GPT_ROOT / "data" / "Osho" / "gpt2"
    device: str = "cpu"
    pretrained_model: str = "gpt2"


def load_settings() -> Settings:
    return Settings(
        scratch_checkpoint=_path_from_env(
            "COMPARE_SCRATCH_CHECKPOINT", Settings.scratch_checkpoint
        ),
        continued_checkpoint=_path_from_env(
            "COMPARE_CONTINUED_CHECKPOINT", Settings.continued_checkpoint
        ),
        lora_checkpoint=_path_from_env(
            "COMPARE_LORA_CHECKPOINT", Settings.lora_checkpoint
        ),
        tokenizer_path=_path_from_env(
            "COMPARE_TOKENIZER_PATH", Settings.tokenizer_path
        ),
        device=os.getenv("COMPARE_DEVICE", Settings.device),
        pretrained_model=os.getenv(
            "COMPARE_PRETRAINED_MODEL", Settings.pretrained_model
        ),
    )

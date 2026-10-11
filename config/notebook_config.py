"""Randomized configuration values for the notebook experiment.

This file collects the main hyperparameters and input-shape values used by the
notebook's attention/model example in a single place.
"""

from dataclasses import dataclass


@dataclass
class NotebookConfig:
    # data / input shape
    batch_size: int = 10
    sequence_length: int = 50
    embedding_dim: int = 256

    # model architecture
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = True
    device: str = "cuda"


# Default instance for quick import
CONFIG = NotebookConfig()

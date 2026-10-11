import os
from pprint import pprint

import torch



ckpt_path = os.path.join("out", "ckpt_scratch_full.pt")

if not os.path.exists(ckpt_path):
    print(f"Checkpoint not found at: {ckpt_path}")
    raise SystemExit(0)

try:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(ckpt_path, map_location="cpu")

print(f"Checkpoint file: {ckpt_path}")
print("Top-level keys:")
print(list(checkpoint.keys()))

print("\nmodel_args:")
pprint(checkpoint.get("model_args", {}))

print("\niter_num:", checkpoint.get("iter_num"))
print("best_val_loss:", checkpoint.get("best_val_loss"))

model_state = checkpoint.get("model", {})
print(f"\nmodel state_dict entries: {len(model_state)}")

for name, value in list(model_state.items())[:10]:
    if isinstance(value, torch.Tensor):
        print(f"{name}: shape={tuple(value.shape)}, dtype={value.dtype}")
    else:
        print(f"{name}: {type(value).__name__}")

if len(model_state) > 10:
    print("...")

optimizer_state = checkpoint.get("optimizer", {})
print("\noptimizer keys:", list(optimizer_state.keys()))

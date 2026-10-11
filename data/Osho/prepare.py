
import os
# import requests
# import tiktoken
import numpy as np
from transformers import GPT2TokenizerFast



input_file_path = r"C:\Abishek\NanoGPT\nanoGPT\Dataset_text.txt"
print(input_file_path)

if not os.path.exists(input_file_path):
    print("File does not exist")
    exit()


with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

tokenizer = GPT2TokenizerFast.from_pretrained(
    r"C:\Abishek\NanoGPT\model\gpt2",
    local_files_only=True
)
# encode with tiktoken gpt2 bpe
# enc = tiktoken.get_encoding("gpt2")
train_ids = tokenizer.encode(train_data)
val_ids = tokenizer.encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens

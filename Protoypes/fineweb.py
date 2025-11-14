"""
FineWeb-Edu tokenizer (low-overhead streaming version)
Produces the same *.npy* shards as the full Karpathy script but without disk explosion.
"""

import os, numpy as np, tiktoken
from datasets import load_dataset
from tqdm import tqdm

# --- Config ---
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"            # same subset
shard_size = int(1e8)                  # 100M tokens per shard (reduce if limited disk)
os.makedirs(local_dir, exist_ok=True)

# --- Load dataset in streaming mode (no local Arrow cache) ---
fw = load_dataset("HuggingFaceFW/fineweb-edu",
                  name=remote_name,
                  split="train",
                  streaming=True)

# --- Tokenizer setup ---
enc = tiktoken.get_encoding("gpt2")
EOT = enc._special_tokens['<|endoftext|>']

def encode_doc(text: str) -> np.ndarray:
    """Encode one document to uint16 token IDs."""
    toks = [EOT]
    toks.extend(enc.encode_ordinary(text))
    arr = np.array(toks, dtype=np.uint32)
    if arr.max() >= 2**16:
        raise ValueError("Token id overflow for uint16.")
    return arr.astype(np.uint16)

# --- Stream through docs and build shards ---
buf = np.empty((shard_size,), dtype=np.uint16)
count, shard_idx = 0, 0
pbar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_idx}")

def flush_shard(n):
    """Write current buffer of n tokens to disk."""
    split = "val" if shard_idx == 0 else "train"
    fn = os.path.join(local_dir, f"edufineweb_{split}_{shard_idx:06d}.npy")
    np.save(fn, buf[:n])
    print(f"Wrote {fn} ({n:,} tokens)")
    return 0  # reset counter

for doc in fw:
    toks = encode_doc(doc["text"])
    i = 0
    while i < len(toks):
        remain = shard_size - count
        take = min(remain, len(toks) - i)
        buf[count:count+take] = toks[i:i+take]
        count += take
        i += take
        pbar.update(take)
        if count == shard_size:
            pbar.close()
            count = flush_shard(count)
            shard_idx += 1
            pbar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_idx}")

# Write last partial shard
if count:
    pbar.close()
    flush_shard(count)

import os
import json
import time
import requests

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from tqdm import tqdm
import tiktoken

# ============================================================
#                    MODEL DEFINITION (MLA GPT)
# ============================================================

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class MLASelfAttention(nn.Module):
    def __init__(self, config, latent_dim_ratio: float = 0.5):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Queries from full embedding
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Latent compression for K/V
        d_latent = int(config.n_embd * latent_dim_ratio)
        self.d_latent = d_latent
        self.c_proj = nn.Linear(config.n_embd, d_latent, bias=False)
        self.k_from_c = nn.Linear(d_latent, config.n_embd, bias=False)
        self.v_from_c = nn.Linear(d_latent, config.n_embd, bias=False)

        # Output projection
        self.c_proj_out = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj_out.NANOGPT_SCALE_INIT = 1

        # Causal mask buffer (not used directly because we use is_causal=True,
        # but kept for compatibility / future use)
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones(config.block_size, config.block_size, dtype=torch.bool)
            ).view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    def forward(self, x, past_key_value=None, use_cache: bool = False):
        """
        x: (B, T, C)
        past_key_value: optional tuple (past_k, past_v) with shapes
                        (B, n_head, T_past, head_dim)
        If use_cache=True, returns (y, (k, v)), otherwise (y, None).
        """
        B, T, C = x.size()
        hs = self.head_dim

        # Q from full embeddings
        q = self.q_proj(x)  # (B, T, C)

        # Latent compression for K/V, then back to full C
        c_latent = self.c_proj(x)          # (B, T, d_latent)
        k_full = self.k_from_c(c_latent)   # (B, T, C)
        v_full = self.v_from_c(c_latent)   # (B, T, C)

        # Reshape to heads: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)      # (B, n_head, T, hs)
        k_full = k_full.view(B, T, self.n_head, hs).transpose(1, 2)
        v_full = v_full.view(B, T, self.n_head, hs).transpose(1, 2)

        if past_key_value is not None:
            past_k, past_v = past_key_value  # (B, n_head, T_past, hs)
            k = torch.cat([past_k, k_full], dim=2)  # (B, n_head, T_past+T, hs)
            v = torch.cat([past_v, v_full], dim=2)
        else:
            k, v = k_full, v_full

        # Multi-head attention with causal masking
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Merge heads back: (B, n_head, T, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj_out(y)

        present_kv = (k, v) if use_cache else None
        return y, present_kv

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MLASelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, past_key_value=None, use_cache: bool = False):
        """
        past_key_value is this block's (k, v) or None.
        Returns (x, present_key_value) if use_cache, else (x, None).
        """
        attn_out, present_kv = self.attn(self.ln_1(x),
                                         past_key_value=past_key_value,
                                         use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx,
        targets=None,
        past_key_values=None,
        use_cache: bool = False,
    ):
        """
        idx: (B, T_new)

        past_key_values: optional list of length n_layer, each element:
            (past_k, past_v) with shape (B, n_head, T_past, head_dim)

        If use_cache=False (training / eval on full sequences):
            returns (logits, loss)

        If use_cache=True (generation with KV cache):
            returns (logits, loss, new_past_key_values)
        """
        B, T = idx.size()
        device = idx.device

        if past_key_values is None:
            past_length = 0
        else:
            # Use first layer to infer past length
            past_length = past_key_values[0][0].size(2)

        assert (
            T + past_length <= self.config.block_size
        ), f"Cannot forward sequence of length {T + past_length}, block size is only {self.config.block_size}"

        # token + positional embeddings
        pos = torch.arange(
            past_length, past_length + T, dtype=torch.long, device=device
        )  # (T,)
        pos_emb = self.transformer.wpe(pos)          # (T, C)
        tok_emb = self.transformer.wte(idx)          # (B, T, C)
        x = tok_emb + pos_emb                        # (B, T, C)  (pos_emb broadcasts)

        new_past_key_values = [] if use_cache else None

        # transformer blocks
        for i, block in enumerate(self.transformer.h):
            past_kv_i = None if past_key_values is None else past_key_values[i]
            x, present_kv = block(
                x,
                past_key_value=past_kv_i,
                use_cache=use_cache,
            )
            if use_cache:
                new_past_key_values.append(present_kv)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        if use_cache:
            return logits, loss, new_past_key_values
        else:
            return logits, loss

# ============================================================
#                TOKENIZER / HELPER FOR GENERATION
# ============================================================

enc = tiktoken.get_encoding("gpt2")

def encode(text: str, device: str) -> torch.Tensor:
    ids = enc.encode(text)
    return torch.tensor([ids], dtype=torch.long, device=device)

def decode(tokens: torch.Tensor) -> str:
    if tokens.dim() == 2:
        tokens = tokens[0]
    return enc.decode(tokens.tolist())

@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
):
    """
    KV-cache aware generation using your MLA model.

    - First runs a full forward pass on the prompt to build the cache.
    - Then generates one token at a time, reusing cached K/V instead of
      recomputing the whole prefix every step.
    """
    model.eval()
    device = idx.device

    # Initial pass over the whole prompt to build caches
    logits, _, past_key_values = model(idx, use_cache=True)

    for _ in range(max_new_tokens):
        # Take logits for the last position
        logits_step = logits[:, -1, :] / max(temperature, 1e-8)

        if top_k is not None:
            v, _ = torch.topk(logits_step, top_k)
            logits_step[logits_step < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits_step, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
        idx = torch.cat((idx, next_token), dim=1)

        # One-step decode with KV cache
        logits, _, past_key_values = model(
            next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )

    return idx

def generate_text(
    model,
    device,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> str:
    x = encode(prompt, device)
    y = generate(
        model,
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return decode(y)

# ============================================================
#                HELLA SWAG EVALUATION SETUP
# ============================================================

try:
    _base_dir = os.path.dirname(__file__)
except NameError:
    _base_dir = os.getcwd()
DATA_CACHE_DIR = os.path.join(_base_dir, "hellaswag")

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val":   "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test":  "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

def download_file(url, fname):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(1024):
            f.write(chunk)
            bar.update(len(chunk))

def download(split):
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    path = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(path):
        print(f"Downloading {split} split...")
        download_file(hellaswags[split], path)

def iterate_examples(split):
    download(split)
    path = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)

def render_example(example):
    """
    Exactly the same logic as the HF script:
    - ctx + each ending
    - mask marks only the ending tokens as 1
    """
    ctx = example["ctx"]
    label = example["label"]
    ends = example["endings"]

    ctx_toks = enc.encode(ctx)

    tok_rows, mask_rows = [], []
    for e in ends:
        e_toks = enc.encode(" " + e)
        tok_rows.append(ctx_toks + e_toks)
        mask_rows.append([0] * len(ctx_toks) + [1] * len(e_toks))

    max_len = max(len(r) for r in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)

    for i, (t, m) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        mask[i, :len(m)] = torch.tensor(m, dtype=torch.long)

    return tokens, mask, label

# ============================================================
#                     LOAD MLA CHECKPOINT
# ============================================================

CKPT_PATH = "step019535.pt"  # change if needed

def load_mla_model(device: str) -> GPT:
    print(f"Loading MLA checkpoint from {CKPT_PATH}...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model

# ============================================================
#                 EVALUATION FUNCTIONS (FIRST 500)
# ============================================================

@torch.no_grad()
def evaluate_mla_on_subset(model: GPT, device: str, max_examples: int = 500):
    """
    Evaluate MLA GPT on the first `max_examples` HellaSwag val examples.
    Returns (acc_raw, acc_norm, elapsed_seconds).
    """
    num_correct = 0
    num_correct_norm = 0
    num_total = 0

    start_time = time.time()

    for example in iterate_examples("val"):
        if num_total >= max_examples:
            break

        tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # Full-sequence forward (no cache) for eval, same as before
        logits, _ = model(tokens)  # (4, T, vocab)

        # Same shifting logic as HF evaluation
        shift_logits = logits[..., :-1, :].contiguous()   # (4, T-1, V)
        shift_tokens = tokens[..., 1:].contiguous()       # (4, T-1)

        losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_tokens.view(-1),
            reduction="none",
        ).view(tokens.size(0), -1)  # (4, T-1)

        shift_mask = mask[..., 1:].contiguous()  # (4, T-1)
        masked_losses = losses * shift_mask

        # Sum over only ending tokens
        sum_loss = masked_losses.sum(dim=1)               # (4,)
        avg_loss = sum_loss / shift_mask.sum(dim=1)       # (4,)

        pred = sum_loss.argmin().item()       # unnormalized total loss
        pred_norm = avg_loss.argmin().item()  # normalized by length

        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)

        if num_total % 100 == 0:
            acc = num_correct / num_total
            acc_norm = num_correct_norm / num_total
            print(f"[MLA] {num_total:5d} examples | acc: {acc:.4f} | acc_norm: {acc_norm:.4f}")

    elapsed = time.time() - start_time
    acc = num_correct / num_total
    acc_norm = num_correct_norm / num_total

    return acc, acc_norm, elapsed

# HF baseline
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

@torch.no_grad()
def load_hf_model(device: str, model_name: str = "rhysjones/gpt2-124M-edu-fineweb-10B"):
    """
    Load HF GPT-2 baseline. Returns (model, model_name).
    """
    print(f"Loading HF model: {model_name}")
    _ = AutoTokenizer.from_pretrained(model_name)  # not used, but kept for completeness
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, model_name

@torch.no_grad()
def evaluate_hf_on_subset(model, device: str, max_examples: int = 500):
    """
    Evaluate HF model on the first `max_examples` HellaSwag val examples.
    Returns (acc_raw, acc_norm, elapsed_seconds).
    """
    num_correct = 0
    num_correct_norm = 0
    num_total = 0

    start_time = time.time()

    for example in iterate_examples("val"):
        if num_total >= max_examples:
            break

        tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        logits = model(tokens).logits  # (4, T, V)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_tokens = tokens[..., 1:].contiguous()

        losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_tokens.view(-1),
            reduction="none"
        ).view(tokens.size(0), -1)

        shift_mask = mask[..., 1:].contiguous()
        masked_losses = losses * shift_mask

        sum_loss = masked_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)

        pred = sum_loss.argmin().item()        # unnormalized total loss
        pred_norm = avg_loss.argmin().item()   # normalized by length

        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)

        if num_total % 100 == 0:
            acc = num_correct / num_total
            acc_norm = num_correct_norm / num_total
            print(f"[HF ] {num_total:5d} examples | acc: {acc:.4f} | acc_norm: {acc_norm:.4f}")

    elapsed = time.time() - start_time
    acc = num_correct / num_total
    acc_norm = num_correct_norm / num_total

    return acc, acc_norm, elapsed

# ============================================================
#                 FORWARD-ONLY SPEED BENCHMARK
# ============================================================

@torch.no_grad()
def benchmark_model_forward(
    name: str,
    model,
    device: str,
    seq_len: int = 512,
    batch_size: int = 8,
    steps: int = 50,
    vocab_size: int | None = None,
):
    """
    Measure average forward-pass latency for a model on random input.
    """
    model.eval()

    if vocab_size is None:
        # Try to infer vocab_size in a generic way
        if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
            vocab_size = model.config.vocab_size
        else:
            vocab_size = model.get_input_embeddings().weight.size(0)

    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )

    # Warmup
    warmup_steps = 10
    for _ in range(warmup_steps):
        _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()
    elif hasattr(torch, "mps") and device == "mps":
        torch.mps.synchronize()

    start = time.time()
    for _ in range(steps):
        _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elif hasattr(torch, "mps") and device == "mps":
        torch.mps.synchronize()
    end = time.time()

    total_time = end - start
    avg_step_time = total_time / steps
    tokens_per_step = batch_size * seq_len
    tokens_per_second = tokens_per_step / avg_step_time

    print(f"\n[Benchmark: {name}]")
    print(f"  seq_len       = {seq_len}")
    print(f"  batch_size    = {batch_size}")
    print(f"  steps         = {steps}")
    print(f"  avg step time = {avg_step_time:.6f} s")
    print(f"  tokens / step = {tokens_per_step}")
    print(f"  tokens / sec  = {tokens_per_second:,.0f}")

# ============================================================
#                           MAIN
# ============================================================

def pick_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    return device

def main():
    max_examples = 500

    device = pick_device()

    hf_model, hf_name = load_hf_model(device)
    mla_model = load_mla_model(device)

    # ------------------------------------------------------------
    # Forward-only benchmark (no HellaSwag loop, same B/T/steps)
    # ------------------------------------------------------------
    print("\n======= FORWARD-ONLY SPEED BENCHMARK (RANDOM TOKENS) =======")
    seq_len = 512
    batch_size = 8
    steps = 50

    # MLA (your GPT)
    benchmark_model_forward(
        name=f"MLA ({CKPT_PATH})",
        model=mla_model,
        device=device,
        seq_len=seq_len,
        batch_size=batch_size,
        steps=steps,
        vocab_size=mla_model.config.vocab_size,
    )

    # HF baseline
    benchmark_model_forward(
        name=f"{hf_name}",
        model=hf_model,
        device=device,
        seq_len=seq_len,
        batch_size=batch_size,
        steps=steps,
        vocab_size=hf_model.config.vocab_size,
    )

if __name__ == "__main__":
    main()

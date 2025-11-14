# infer_mla.py
import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import argparse
import tiktoken


# ----------------- Model definitions (same as training) ----------------- #

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

        # Q projection (full dim)
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

        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones(config.block_size, config.block_size, dtype=torch.bool)
            ).view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.size()
        hs = self.head_dim

        q = self.q_proj(x)
        c_latent = self.c_proj(x)
        k = self.k_from_c(c_latent)
        v = self.v_from_c(c_latent)

        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, h, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj_out(y)
        return y


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

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


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

        # weight tying
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

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)          # (T, C)
        tok_emb = self.transformer.wte(idx)          # (B, T, C)
        x = tok_emb + pos_emb                        # broadcast over B

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ----------------- Sampling utilities ----------------- #

def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Autoregressively generate tokens from the model.

    idx: (1, T) LongTensor with starting tokens.
    """
    model.eval()
    device = next(model.parameters()).device

    for _ in range(max_new_tokens):
        # crop to block size
        idx_cond = idx[:, -model.config.block_size :]

        with torch.no_grad():
            logits, _ = model(idx_cond)

        logits = logits[:, -1, :] / max(temperature, 1e-8)

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)

    return idx


# ----------------- Main CLI entry ----------------- #

def main():
    parser = argparse.ArgumentParser(description="Run inference with MLA GPT checkpoint.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="step019535.pt",
        help="Path to checkpoint (can be full payload or state_dict).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, my name is",
        help="Prompt text to start generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling (set to 0 to disable).",
    )

    args = parser.parse_args()

    # ----- Device ----- #
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # ----- Load checkpoint ----- #
    print(f"Loading checkpoint from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=device)

    # Handle both "full payload" and "raw state_dict"
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        cfg_dict = ckpt.get("config", {})
        config = GPTConfig(**cfg_dict)
    else:
        state_dict = ckpt
        # default to your training config (GPT-2 124M with padded vocab)
        config = GPTConfig(vocab_size=50304)

    # ----- Build model & load weights ----- #
    model = GPT(config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # ----- Tokenizer ----- #
    enc = tiktoken.get_encoding("gpt2")

    # Encode prompt
    prompt_ids = enc.encode(args.prompt)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # ----- Generate ----- #
    out_ids = generate(
        model,
        prompt_tensor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=(None if args.top_k <= 0 else args.top_k),
    )

    # Decode
    out_text = enc.decode(out_ids[0].tolist())
    print("\n================== GENERATED TEXT ==================\n")
    print(out_text)
    print("\n====================================================\n")


if __name__ == "__main__":
    main()

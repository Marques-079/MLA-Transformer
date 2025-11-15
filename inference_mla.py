import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import tiktoken

# ---------------------- Model definition ----------------------

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

        # Queries from full emb
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
        # but kept here for compatibility / future use)
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

        # Q from full embeddings
        q = self.q_proj(x)

        # Latent compression for K/V
        c_latent = self.c_proj(x)
        k = self.k_from_c(c_latent)
        v = self.v_from_c(c_latent)

        # Reshape to heads: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        # Multi-head attention with causal masking
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Merge heads back: (B, n_head, T, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj_out(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
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

    def forward(self, idx, targets=None):
        # idx: (B, T)
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # token + positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        pos_emb = self.transformer.wpe(pos)  # (T, C)
        tok_emb = self.transformer.wte(idx)  # (B, T, C)
        x = tok_emb + pos_emb  # broadcast pos_emb over batch

        # transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

# ---------------------- Inference utilities ----------------------

# Device selection
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Load checkpoint (change path as needed)
CKPT_PATH = "step019535.pt"  # or "checkpoints/step00xxxx.pt"
ckpt = torch.load(CKPT_PATH, map_location="cpu")

# Rebuild config from checkpoint
config = GPTConfig(**ckpt["config"])

# Build model and load weights
model = GPT(config)
model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()

print(f"Loaded model from {CKPT_PATH}")

# Tokenizer
enc = tiktoken.get_encoding("gpt2")

def encode(text: str) -> torch.Tensor:
    ids = enc.encode(text)
    return torch.tensor([ids], dtype=torch.long, device=device)

def decode(tokens: torch.Tensor) -> str:
    if tokens.dim() == 2:
        tokens = tokens[0]
    return enc.decode(tokens.tolist())

@torch.no_grad()
def generate(model, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
    """
    idx: (B, T) starting tokens
    returns: (B, T + max_new_tokens)
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
        idx = torch.cat((idx, next_token), dim=1)

    return idx

def generate_text(
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> str:
    x = encode(prompt)
    y = generate(model, x, max_new_tokens=max_new_tokens,
                 temperature=temperature, top_k=top_k)
    return decode(y)

if __name__ == "__main__":
    prompt = "How many brains does an octopus have?"
    print("=== PROMPT ===")
    print(prompt)
    print("\n=== COMPLETION ===")
    print(generate_text(prompt))

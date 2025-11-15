import os
import json
import time
import math
import requests

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from tqdm import tqdm
import tiktoken
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

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
    model,
    idx,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
):
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


# HF baseline loader
@torch.no_grad()
def load_hf_model(
    device: str,
    model_name: str = "rhysjones/gpt2-124M-edu-fineweb-10B",
):
    """
    Load HF GPT-2-type baseline. Returns (model, model_name).
    """
    print(f"Loading HF model: {model_name}")
    _ = AutoTokenizer.from_pretrained(model_name)  # not used, but kept for completeness
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, model_name


# ============================================================
#              GENERIC LOGIT HELPERS / NLL UTILS
# ============================================================

@torch.no_grad()
def forward_logits(model, tokens: torch.Tensor) -> torch.Tensor:
    """
    Call either our GPT (returns (logits, loss)) or HF model (returns obj.logits)
    and always get back the logits tensor of shape (B, T, V).
    """
    out = model(tokens)
    # Our GPT returns (logits, loss)
    if isinstance(out, tuple):
        logits = out[0]
    else:
        # HF models return a ModelOutput with .logits
        logits = out.logits
    return logits


@torch.no_grad()
def compute_nll_for_sequence(
    model,
    device: str,
    token_ids: torch.Tensor,
    block_size: int | None = None,
) -> tuple[float, int]:
    """
    Compute total negative log-likelihood and token count for a 1D tensor of token ids.
    We evaluate in sliding windows of length `block_size` for memory safety.
    """
    token_ids = token_ids.to("cpu")
    if block_size is None and hasattr(model, "config") and hasattr(
        model.config, "block_size"
    ):
        block_size = model.config.block_size
    if block_size is None:
        block_size = 1024

    total_nll = 0.0
    total_tokens = 0

    # We predict token t given all tokens < t
    # So for window [start : end+1] we input ids[start:end] and predict ids[start+1:end+1]
    for start in range(0, token_ids.size(0) - 1, block_size):
        end = min(start + block_size, token_ids.size(0) - 1)
        ctx = token_ids[start:end]  # length L
        targets = token_ids[start + 1 : end + 1]  # length L

        if ctx.numel() < 1 or targets.numel() < 1:
            continue

        ctx = ctx.unsqueeze(0).to(device)  # (1, L)
        targets = targets.unsqueeze(0).to(device)  # (1, L)

        logits = forward_logits(model, ctx)  # (1, L, V)
        shift_logits = logits[..., :-1, :].contiguous()  # (1, L-1, V)
        shift_targets = targets[..., : shift_logits.size(1)]

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            reduction="sum",
        )
        total_nll += loss.item()
        total_tokens += shift_targets.numel()

    return total_nll, total_tokens


# ============================================================
#                 1) WIKITEXT-2 PERPLEXITY
# ============================================================

@torch.no_grad()
def evaluate_wikitext_perplexity(
    model,
    device: str,
    split: str = "validation",
    max_documents: int | None = None,
):
    """
    Compute standard LM perplexity on WikiText-2 (Salesforce/wikitext, wikitext-2-raw-v1).
    """
    print(f"\n[WIKITEXT-2] Evaluating perplexity on split='{split}'...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    total_nll = 0.0
    total_tokens = 0

    for i, ex in enumerate(tqdm(ds, desc="WIKITEXT-2 docs")):
        if max_documents is not None and i >= max_documents:
            break
        text = ex["text"]
        if not text or not text.strip():
            continue

        ids = enc.encode(text)
        if len(ids) < 2:
            continue

        token_ids = torch.tensor(ids, dtype=torch.long)
        nll, n_tok = compute_nll_for_sequence(model, device, token_ids)
        total_nll += nll
        total_tokens += n_tok

    if total_tokens == 0:
        ppl = float("inf")
    else:
        ppl = math.exp(total_nll / total_tokens)
    print(f"[WIKITEXT-2] total tokens: {total_tokens}, perplexity: {ppl:.3f}")
    return ppl


# ============================================================
#          2) LAMBADA (LAST-WORD PREDICTION ACCURACY)
# ============================================================

@torch.no_grad()
def evaluate_lambada_last_word_accuracy(
    model,
    device: str,
    split: str = "test",
    max_examples: int | None = None,
):
    """
    LAMBADA (EleutherAI/lambada_openai).

    Dataset only has a 'test' split, so we default to that.
    Metric: accuracy on predicting the final token of each passage
    given all previous tokens.
    """
    print(f"\n[LAMBADA] Evaluating last-word accuracy on split='{split}'...")
    ds = load_dataset("EleutherAI/lambada_openai", split=split)

    num_correct = 0
    num_total = 0

    for i, ex in enumerate(tqdm(ds, desc="LAMBADA examples")):
        if max_examples is not None and i >= max_examples:
            break
        text = ex["text"]
        ids = enc.encode(text)
        if len(ids) < 2:
            continue

        context_ids = torch.tensor(
            ids[:-1],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        target_id = ids[-1]

        logits = forward_logits(model, context_ids)  # (1, T, V)
        last_logits = logits[:, -1, :]
        pred_id = int(last_logits.argmax(dim=-1)[0].item())

        num_total += 1
        num_correct += int(pred_id == target_id)

    acc = num_correct / num_total if num_total > 0 else 0.0
    print(f"[LAMBADA] accuracy: {acc:.4f} ({num_correct}/{num_total})")
    return acc


# ============================================================
#      3) PIQA (PHYSICAL COMMONSENSE MULTIPLE-CHOICE ACC)
# ============================================================

def build_mc_tokens_and_mask(context: str, candidates: list[str]):
    """
    Generic helper for MC-style datasets:
      - context: shared prefix string
      - candidates: list of answer completions

    Returns:
      tokens: (N, T_max) int64 tensor
      mask:   (N, T_max) int64 tensor, 1s over candidate tokens only
    """
    ctx_ids = enc.encode(context)
    tok_rows, mask_rows = [], []

    for c in candidates:
        c_ids = enc.encode(" " + c)
        tok_rows.append(ctx_ids + c_ids)
        mask_rows.append([0] * len(ctx_ids) + [1] * len(c_ids))

    max_len = max(len(r) for r in tok_rows)
    tokens = torch.zeros((len(candidates), max_len), dtype=torch.long)
    mask = torch.zeros((len(candidates), max_len), dtype=torch.long)

    for i, (t, m) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, : len(t)] = torch.tensor(t, dtype=torch.long)
        mask[i, : len(m)] = torch.tensor(m, dtype=torch.long)

    return tokens, mask


@torch.no_grad()
def mc_choice_from_masked_nll(
    model,
    device: str,
    tokens: torch.Tensor,
    mask: torch.Tensor,
) -> int:
    """
    Given a batch of MC options encoded as:
      tokens: (N, T)
      mask:   (N, T)  (1 where the answer tokens are)
    returns the index (0..N-1) of the option with the lowest
    length-normalized masked loss.
    """
    tokens = tokens.to(device)
    mask = mask.to(device)

    logits = forward_logits(model, tokens)  # (N, T, V)

    shift_logits = logits[..., :-1, :].contiguous()  # (N, T-1, V)
    shift_tokens = tokens[..., 1:].contiguous()  # (N, T-1)
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_tokens.view(-1),
        reduction="none",
    ).view(tokens.size(0), -1)  # (N, T-1)

    shift_mask = mask[..., 1:].contiguous()  # (N, T-1)
    masked_losses = losses * shift_mask

    sum_loss = masked_losses.sum(dim=1)  # (N,)
    denom = shift_mask.sum(dim=1).clamp(min=1)
    avg_loss = sum_loss / denom  # (N,)

    pred = int(avg_loss.argmin().item())
    return pred


@torch.no_grad()
def evaluate_piqa_accuracy(
    model,
    device: str,
    max_examples: int | None = None,
):
    """
    PIQA: commonsense physical reasoning
      - Dataset: ybisk/piqa (default)
      - Metric: accuracy (normalized masked NLL over solutions)
    """
    print("\n[PIQA] Evaluating multiple-choice accuracy...")
    ds = load_dataset("ybisk/piqa", split="validation")

    num_correct = 0
    num_total = 0

    for i, ex in enumerate(tqdm(ds, desc="PIQA examples")):
        if max_examples is not None and i >= max_examples:
            break

        context = ex["goal"]
        candidates = [ex["sol1"], ex["sol2"]]
        label = int(ex["label"])  # 0 or 1

        tokens, mask = build_mc_tokens_and_mask(context, candidates)
        pred = mc_choice_from_masked_nll(model, device, tokens, mask)

        num_total += 1
        num_correct += int(pred == label)

    acc = num_correct / num_total if num_total > 0 else 0.0
    print(f"[PIQA] accuracy: {acc:.4f} ({num_correct}/{num_total})")
    return acc


# ============================================================
#   4) WINOGRANDE (COMMONSENSE PRONOUN/CLOZE MULTIPLE-CHOICE)
# ============================================================

@torch.no_grad()
def evaluate_winogrande_accuracy(
    model,
    device: str,
    subset: str = "winogrande_l",
    max_examples: int | None = None,
):
    """
    Winogrande: pronoun/cloze-style commonsense reasoning
      - Dataset: allenai/winogrande
      - Config: subset (e.g. 'winogrande_l')
      - Metric: accuracy (normalized masked NLL over each filled-in sentence)
    """
    print(f"\n[Winogrande-{subset}] Evaluating multiple-choice accuracy...")
    ds = load_dataset("allenai/winogrande", subset, split="validation")

    num_correct = 0
    num_total = 0

    for i, ex in enumerate(tqdm(ds, desc="Winogrande examples")):
        if max_examples is not None and i >= max_examples:
            break

        sentence = ex["sentence"]
        option1 = ex["option1"]
        option2 = ex["option2"]
        answer = ex["answer"]  # "1" or "2"

        # Replace the underscore with each option
        cand1 = sentence.replace("_", option1)
        cand2 = sentence.replace("_", option2)
        candidates = [cand1, cand2]

        # Empty context, mask entire candidate sentences
        tokens, mask = build_mc_tokens_and_mask("", candidates)
        pred = mc_choice_from_masked_nll(model, device, tokens, mask)

        gold = 0 if answer == "1" else 1

        num_total += 1
        num_correct += int(pred == gold)

    acc = num_correct / num_total if num_total > 0 else 0.0
    print(f"[Winogrande-{subset}] accuracy: {acc:.4f} ({num_correct}/{num_total})")
    return acc


# ============================================================
#   5) HELLASWAG (KARPATHY-STYLE MC EVAL, MASKED NLL)
# ============================================================

@torch.no_grad()
def evaluate_hellaswag_accuracy(
    model,
    device: str,
    split: str = "validation",
    max_examples: int | None = None,
):
    """
    HellaSwag multiple-choice evaluation, in the style of Karpathy:
      - context = ctx_a + ctx_b
      - candidates = endings (4 options)
      - score = length-normalized masked NLL over the ending tokens
    Dataset: 'hellaswag' on HF.
    """
    print(f"\n[HellaSwag] Evaluating accuracy on split='{split}'...")
    ds = load_dataset("hellaswag", split=split)

    num_correct = 0
    num_total = 0

    for i, ex in enumerate(tqdm(ds, desc="HellaSwag examples")):
        if max_examples is not None and i >= max_examples:
            break

        ctx_a = ex["ctx_a"]
        ctx_b = ex["ctx_b"]
        context = ctx_a + " " + ctx_b
        candidates = ex["endings"]  # list of 4 strings
        label = int(ex["label"])    # 0..3

        tokens, mask = build_mc_tokens_and_mask(context, candidates)
        pred = mc_choice_from_masked_nll(model, device, tokens, mask)

        num_total += 1
        num_correct += int(pred == label)

    acc = num_correct / num_total if num_total > 0 else 0.0
    print(f"[HellaSwag] accuracy: {acc:.4f} ({num_correct}/{num_total})")
    return acc


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

    - name: label to print (e.g. "MLA" or "gpt2-124M-edu-fineweb-10B")
    - model: callable model (either your GPT or HF AutoModelForCausalLM)
    - device: "cuda", "mps", or "cpu"
    - seq_len: sequence length T
    - batch_size: batch size B
    - steps: number of timed forward passes
    - vocab_size: vocab size for random token generation
                  (if None, attempts to infer from model)

    Returns:
      (avg_step_time, tokens_per_second)
    """
    model.eval()

    if vocab_size is None:
        # Try to infer vocab_size in a generic way
        if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
            vocab_size = model.config.vocab_size
        else:
            vocab_size = model.get_input_embeddings().weight.size(0)

    # Random integer tokens in [0, vocab_size)
    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )

    # Warmup: fill caches, JIT, etc.
    warmup_steps = 10
    for _ in range(warmup_steps):
        _ = model(x)

    # Ensure all previous work is done before timing
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()

    start = time.time()
    for _ in range(steps):
        _ = model(x)
    # Sync again so we don't stop the timer early
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
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

    return avg_step_time, tokens_per_second


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
    device = pick_device()

    # Load models
    hf_fineweb_model, hf_fineweb_name = load_hf_model(
        device,
        model_name="rhysjones/gpt2-124M-edu-fineweb-10B",
    )
    hf_gpt2_model, hf_gpt2_name = load_hf_model(
        device,
        model_name="openai-community/gpt2",
    )
    mla_model = load_mla_model(device)

    # Container for metrics to write to file later
    results = {}

    # Common settings
    subset = "winogrande_l"
    seq_len = 512
    batch_size = 8
    steps = 50

    # ----------------- MLA METRICS -----------------
    print("\n==================== MLA MODEL ====================")
    mla_wt2_ppl = evaluate_wikitext_perplexity(
        mla_model,
        device,
        split="validation",
    )
    mla_lambada_acc = evaluate_lambada_last_word_accuracy(mla_model, device)
    mla_piqa_acc = evaluate_piqa_accuracy(mla_model, device)
    mla_wg_acc = evaluate_winogrande_accuracy(
        mla_model,
        device,
        subset=subset,
    )
    mla_hellaswag_acc = evaluate_hellaswag_accuracy(
        mla_model,
        device,
        split="validation",
    )

    mla_avg_step_time, mla_tokens_per_sec = benchmark_model_forward(
        name=f"MLA ({CKPT_PATH})",
        model=mla_model,
        device=device,
        seq_len=seq_len,
        batch_size=batch_size,
        steps=steps,
        vocab_size=mla_model.config.vocab_size,
    )

    results["MLA"] = {
        "description": f"MLA checkpoint {CKPT_PATH}",
        "wikitext2_ppl_val": mla_wt2_ppl,
        "lambada_acc": mla_lambada_acc,
        "piqa_acc": mla_piqa_acc,
        "winogrande_acc": mla_wg_acc,
        "hellaswag_acc": mla_hellaswag_acc,
        "forward_avg_step_time": mla_avg_step_time,
        "forward_tokens_per_second": mla_tokens_per_sec,
        "forward_seq_len": seq_len,
        "forward_batch_size": batch_size,
    }

    # ----------------- HF FINEWEB MODEL -----------------
    print("\n==================== HF FINEWEB MODEL ====================")
    hf_wt2_ppl = evaluate_wikitext_perplexity(
        hf_fineweb_model,
        device,
        split="validation",
    )
    hf_lambada_acc = evaluate_lambada_last_word_accuracy(
        hf_fineweb_model,
        device,
    )
    hf_piqa_acc = evaluate_piqa_accuracy(hf_fineweb_model, device)
    hf_wg_acc = evaluate_winogrande_accuracy(
        hf_fineweb_model,
        device,
        subset=subset,
    )
    hf_hellaswag_acc = evaluate_hellaswag_accuracy(
        hf_fineweb_model,
        device,
        split="validation",
    )

    hf_avg_step_time, hf_tokens_per_sec = benchmark_model_forward(
        name=f"{hf_fineweb_name}",
        model=hf_fineweb_model,
        device=device,
        seq_len=seq_len,
        batch_size=batch_size,
        steps=steps,
        vocab_size=hf_fineweb_model.config.vocab_size,
    )

    results[hf_fineweb_name] = {
        "description": hf_fineweb_name,
        "wikitext2_ppl_val": hf_wt2_ppl,
        "lambada_acc": hf_lambada_acc,
        "piqa_acc": hf_piqa_acc,
        "winogrande_acc": hf_wg_acc,
        "hellaswag_acc": hf_hellaswag_acc,
        "forward_avg_step_time": hf_avg_step_time,
        "forward_tokens_per_second": hf_tokens_per_sec,
        "forward_seq_len": seq_len,
        "forward_batch_size": batch_size,
    }

    # ----------------- HF GPT2-124M (OPENAI COMMUNITY) -----------------
    print("\n==================== HF GPT2 BASE MODEL ====================")
    gpt2_wt2_ppl = evaluate_wikitext_perplexity(
        hf_gpt2_model,
        device,
        split="validation",
    )
    gpt2_lambada_acc = evaluate_lambada_last_word_accuracy(
        hf_gpt2_model,
        device,
    )
    gpt2_piqa_acc = evaluate_piqa_accuracy(hf_gpt2_model, device)
    gpt2_wg_acc = evaluate_winogrande_accuracy(
        hf_gpt2_model,
        device,
        subset=subset,
    )
    gpt2_hellaswag_acc = evaluate_hellaswag_accuracy(
        hf_gpt2_model,
        device,
        split="validation",
    )

    gpt2_avg_step_time, gpt2_tokens_per_sec = benchmark_model_forward(
        name=f"{hf_gpt2_name}",
        model=hf_gpt2_model,
        device=device,
        seq_len=seq_len,
        batch_size=batch_size,
        steps=steps,
        vocab_size=hf_gpt2_model.config.vocab_size,
    )

    results[hf_gpt2_name] = {
        "description": hf_gpt2_name,
        "wikitext2_ppl_val": gpt2_wt2_ppl,
        "lambada_acc": gpt2_lambada_acc,
        "piqa_acc": gpt2_piqa_acc,
        "winogrande_acc": gpt2_wg_acc,
        "hellaswag_acc": gpt2_hellaswag_acc,
        "forward_avg_step_time": gpt2_avg_step_time,
        "forward_tokens_per_second": gpt2_tokens_per_sec,
        "forward_seq_len": seq_len,
        "forward_batch_size": batch_size,
    }

    # ----------------- SUMMARY PRINT -----------------
    print("\n==================== METRIC SUMMARY ====================")
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        print(f"  WikiText-2 ppl (val):    {metrics['wikitext2_ppl_val']:.3f}")
        print(f"  LAMBADA acc:             {metrics['lambada_acc']:.4f}")
        print(f"  PIQA acc:                {metrics['piqa_acc']:.4f}")
        print(f"  Winogrande-{subset} acc: {metrics['winogrande_acc']:.4f}")
        print(f"  HellaSwag acc:           {metrics['hellaswag_acc']:.4f}")
        print(
            f"  Fwd tokens/sec (B={metrics['forward_batch_size']}, T={metrics['forward_seq_len']}): "
            f"{metrics['forward_tokens_per_second']:,.0f}"
        )
    print("=======================================================\n")

    # ----------------- WRITE RESULTS TO TXT (JSON) -----------------
    output_path = "eval_results.txt"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved all metrics and speed benchmarks to '{output_path}'\n")


if __name__ == "__main__":
    main()

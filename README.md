# MLA-Transformer: Evaluating Multi-Linear Attention vs Multi-Head Attention at GPT-2 Scale

This project implements a GPT-2-124M–scale Transformer with **Multi-Linear Attention (MLA)** and benchmarks it against two standard **Multi-Head Attention (MHA)** baselines:

- `openai-community/gpt2` (GPT-2 124M)
- `rhysjones/gpt2-124M-edu-fineweb-10B` (GPT-2 124M trained on FineWeb-Edu 10B tokens)

The goal is to see **how much of MLA’s theoretical efficiency** (less K/V projection compute) is visible **at small scale** and on consumer hardware, while keeping model size and evaluation settings as comparable as possible.

---

## 1. High-Level Overview

MLA is an attention variant that:

- **Compresses K/V** into a latent dimension `C_latent < C`
- **Re-expands** them back to the full embedding dimension before the attention kernel
- Aims to reduce **K/V projection FLOPs** and memory bandwidth, especially on **large models** and **long contexts**

In this repo:

- I re-implement a **GPT-2-style architecture** in PyTorch with MLA (`MLASelfAttention`).
- I compare it to HuggingFace GPT-2 baselines using:
  - **Language modeling metrics** (perplexity)
  - **Downstream multiple-choice accuracy**
  - **Forward pass throughput (tokens/s)**
  - **Autoregressive generation throughput (tokens/s)**
  - **Time-to-First-Token (TTFT)**
  - **KV-cache generation throughput**

The experiments are run on **Apple M-series (MPS backend)**, so results include real-world hardware constraints, not just FLOP counts.

---

## 2. Figures (Placeholders)

You can drop your own diagrams / plots here. These are just suggested roles.

**Figure 1 – Architecture diagram: MLA vs MHA in a GPT-2 block**

```markdown
![Figure 1 – MLA vs MHA block diagram](images/fig1_architecture.png)

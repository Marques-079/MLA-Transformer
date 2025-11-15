# MLA-Transformer: Evaluating Multi-Linear Attention vs Multi-Head Attention at GPT-2 Scale 🛠️🧠

This project implements a GPT-2-124M–scale Transformer with **Multi-Linear Attention (MLA)** and benchmarks it against two standard **Multi-Head Attention (MHA)** baselines:

- `openai-community/gpt2` (GPT-2 124M)
- `rhysjones/gpt2-124M-edu-fineweb-10B` (GPT-2 124M trained on FineWeb-Edu 10B tokens)

The goal is to see **how much of MLA’s theoretical efficiency** (less K/V projection compute) is visible **at small scale** and on consumer hardware, while keeping model size and evaluation settings as comparable as possible.

---

Traditional Multi-Head Attention (MHA) mechanisms in transformer models store full-sized keys (K) and values (V) in a "KV cache" during text generation. As sequence lengths increase, this cache becomes a major memory bottleneck. MLA, introduced by DeepSeek-V2 in 2024. I want to see if this shows reasonable gains on smaller scales. Improvements made are : 


- **Compresses K/V** into a latent dimension `C_latent < C` 🪢
- **Re-expands** them back to the full embedding dimension before the attention kernel 🔄
- Reduces **K/V projection FLOPs** and memory bandwidth, especially on **large models** and **long contexts** 📉

Disclaimer : Conventionally MLA is applied to large models with specialsied hardware
---

## Image Gallery 📸

<div align="center">
  <div>
    <img src="https://github.com/Marques-079/MLA-Transformer/blob/abcc4b2c26702dd44e2235489be2433811a20f6e/losssesfinalg.png"
         alt="Figure 3"
         width="100%" />
  </div>
  <div>
    <img src="https://github.com/Marques-079/MLA-Transformer/blob/abcc4b2c26702dd44e2235489be2433811a20f6e/trainingimg.png"
         alt="Figure 3"
         width="100%" />
  </div>
</div>

---

## Results on Evals 🧬🔍
```
================ FINAL COMPARISON ================

MLA:
  WikiText2  PPL           : 66.67
  LAMBADA    PPL           : 46.93
  LAMBADA    ACC           : 39.61%
  HellaSwag  ACC           : 30.23%
  Forward    tokens / sec  : 9634.3
  Generation tokens / sec  : 56.9

FineWeb_124M:
  WikiText2  PPL           : 59.46
  LAMBADA    PPL           : 42.47
  LAMBADA    ACC           : 41.84%
  HellaSwag  ACC           : 30.93%
  Forward    tokens / sec  : 8483.6
  Generation tokens / sec  : 54.7

GPT2_124M:
  WikiText2  PPL           : 46.82
  LAMBADA    PPL           : 37.90
  LAMBADA    ACC           : 43.40%
  HellaSwag  ACC           : 30.28%
  Forward    tokens / sec  : 8483.7
  Generation tokens / sec  : 57.1
==================================================
```
---
## What do these show? 🔭

pros: 
- MLA delivers a ~13–14% increase in forward tokens/sec compared to both GPT-2 baselines at the same parameter scale.
- The lowest TTFT among the three models indicates a latency benefit for workloads dominated by full-sequence forward passes (e.g., scoring, log-prob computation)

cons: 
- Perplexity (PPL) was consistently worse than both GPT-2 baselines
- LAMBADA accuracy was 2–4% lower, indicating weaker long-range dependency modeling.
  
---
## Improvements 📝📘

 - **Model scale (124M parameters)**
At GPT-2 scale, attention FLOPs are a relatively small fraction of total compute; MLP layers dominate.
MLA optimizes only the attention component, so the potential wall-clock gain is inherently limited.

 - **Fused vs unfused projections**
GPT-2 baselines use a single fused QKV projection (c_attn) implemented as one large, optimized matmul.
MLA uses multiple separate projections (q_proj, c_proj, k_from_c, v_from_c), leading to more and smaller matmul kernels.
On MPS, many small kernels introduce overhead that can outweigh theoretical FLOP savings.

 - **Hardware backend (Apple MPS)**
The MPS backend is less optimized for transformer workloads than CUDA with FlashAttention/xFormers.
No custom fused MLA kernels, no FlashAttention-style integration, and limited kernel fusion options are available.

 - **KV-cache interaction**
MLA’s K/V compression primarily reduces K/V projection compute when K/V are recomputed frequently.
In KV-cache decoding, K/V are computed once per token and then reused; the savings from compression are less pronounced, while overhead from extra projections remains.

---

## Acknowledgements 📌

GPT-2 architecture inspiration from OpenAI GPT-2 and NanoGPT-style implementations
```
Datasets sourced via HuggingFace Datasets:
wikitext
EleutherAI/lambada_openai
hellaswag
Baseline models:
openai-community/gpt2
rhysjones/gpt2-124M-edu-fineweb-10B
```


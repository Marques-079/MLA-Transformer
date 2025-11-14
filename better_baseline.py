from dataclasses import dataclass
import torch 
import torch.nn as nn
from torch.nn import functional as F 
import math
import time 
import inspect
import os 
import numpy as np

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    
class MLASelfAttention(nn.Module):
    def __init__(self, config, latent_dim_ratio=0.5):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        #encode to lower dimensiosn
        d_latent = int(config.n_embd * latent_dim_ratio)
        self.d_latent = d_latent
        self.c_proj = nn.Linear(config.n_embd, d_latent, bias=False)
        #decode
        self.k_from_c = nn.Linear(d_latent, config.n_embd, bias=False)
        self.v_from_c = nn.Linear(d_latent, config.n_embd, bias=False)

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

        #Normal query forward pass
        q = self.q_proj(x)
        # Latent compression for K/V cache
        c_latent = self.c_proj(x)
        k = self.k_from_c(c_latent)
        v = self.v_from_c(c_latent)

        #reshape tp heads
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
        self.gelu = nn.GELU(approximate = 'tanh')
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

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights) 
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std=0.02)


    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


    #Andrej Optimizer code
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    # #Andrej Autoloader code
    # @classmethod
    # def from_pretrained(cls, model_type):
    #     """Loads pretrained GPT-2 model weights from huggingface"""
    #     assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    #     from transformers import GPT2LMHeadModel
    #     print("loading weights from pretrained gpt: %s" % model_type)

    #     # n_layer, n_head and n_embd are determined from model_type
    #     config_args = {
    #         'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    #         'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
    #         'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
    #         'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    #     }[model_type]
    #     config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
    #     config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
    #     # create a from-scratch initialized minGPT model
    #     config = GPTConfig(**config_args)
    #     model = GPT(config)
    #     sd = model.state_dict()
    #     sd_keys = sd.keys()
    #     sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

    #     # init a huggingface/transformers model
    #     model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    #     sd_hf = model_hf.state_dict()

    #     # copy while ensuring all of the parameters are aligned and match in names and shapes
    #     sd_keys_hf = sd_hf.keys()
    #     sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
    #     sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
    #     transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    #     # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
    #     # this means that we have to transpose these weights when we import them
    #     assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    #     for k in sd_keys_hf:
    #         if any(k.endswith(w) for w in transposed):
    #             # special treatment for the Conv1D weights we need to transpose
    #             assert sd_hf[k].shape[::-1] == sd[k].shape
    #             with torch.no_grad():
    #                 sd[k].copy_(sd_hf[k].t())
    #         else:
    #             # vanilla copy over the other parameters
    #             assert sd_hf[k].shape == sd[k].shape
    #             with torch.no_grad():
    #                 sd[k].copy_(sd_hf[k])

    #     return model

import tiktoken

def load_tokens(filename):
    npt = np.load(filename)
    ppt = torch.tensor(npt, dtype=torch.long)
    return ppt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        assert split in {'train', 'val'}
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()
        
    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y



#----------------------------------------------Main Training Loop------------------------------------------------------------------

# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 main.py

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp = int(os.environ.get('RANK', -1)) != -1 
if ddp:
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # Only set device if not using DDP
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

torch.manual_seed(1337)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(1337)
elif torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

#gradient accumulation
total_batch_size = 524288
B = 64
T = 1024
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"Using gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader   = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

#Create and compile parallel processes + model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

print(f"Using device: {device}")

# Wrap with DDP first, then compile (better compatibility)
if ddp: 
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

if device.startswith("cuda"):  # only CUDA supports compile+DDP well
    model = torch.compile(model, mode="max-autotune")  # or mode="reduce-overhead"
else:
    print("Running eager mode (compile unsupported on non-CUDA)")

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 357
max_steps = 9536
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

#Checkpoint saving with linspacing 
ckpt_dir = "checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

# pick 5 steps evenly from [1, max_steps-1]
save_steps = set(np.linspace(1, max_steps - 1, 10, dtype=int).tolist())

def save_ckpt(step, model, optimizer, extra_tag=None):
    raw = model.module if isinstance(model, DDP) else model
    payload = {
        "step": step,
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": raw.config.__dict__,
    }
    name = f"step{step:06d}.pt" if extra_tag is None else f"{extra_tag}.pt"
    path = os.path.join(ckpt_dir, name)
    torch.save(payload, path)
    if master_process:
        print(f"[ckpt] saved {path}")

def save_weights_only(model, filename):
    raw = model.module if isinstance(model, DDP) else model
    torch.save(raw.state_dict(), os.path.join(ckpt_dir, filename))

# Get device type for autocast and optimizer (cuda, cpu, or mps)
device_type = device.split(':')[0] if ':' in device else device

#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas = (0.9, 0.95), eps = 1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

for step in range(max_steps):
    start = time.perf_counter()

    # Once in a while evaluate our validation loss
    if step % 100 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()

        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")

    model.train()
    optimizer.zero_grad()

    loss_accum = 0.0

    old_w = raw_model.lm_head.weight.clone().detach()
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        #Temporary dtype swap for ops -> Faster computation with minimal loss to precision
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        #Active syncing on last
        if ddp and micro_step != grad_accum_steps - 1:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    end = time.perf_counter()
    # After training step
    new_w = raw_model.lm_head.weight.clone().detach()

    diff = (new_w - old_w).abs().mean()
    time_taken = end - start
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / (end - start)
    if master_process:
        print(f"Step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | tokens/sec: {tokens_per_sec:.2f} | norm: {norm:.2f} | time: {time_taken * 1000:.2f}ms")
    
    # periodic checkpoint: 5 evenly distributed saves
    if master_process and step in save_steps:
        save_ckpt(step, model, optimizer)

if master_process:
    save_ckpt(step, model, optimizer, extra_tag="final")
    save_weights_only(model, "weights_only.pt")

if ddp:
    destroy_process_group()

import sys; sys.exit(0)





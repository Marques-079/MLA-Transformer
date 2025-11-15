from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os, json, requests
from tqdm import tqdm
import tiktoken
from torch.nn import functional as F

# --------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

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

def render_example(example):
    ctx = example["ctx"]; label = example["label"]; ends = example["endings"]
    ctx_toks = enc.encode(ctx)
    tok_rows, mask_rows = [], []
    for e in ends:
        e_toks = enc.encode(" " + e)
        tok_rows.append(ctx_toks + e_toks)
        mask_rows.append([0]*len(ctx_toks) + [1]*len(e_toks))
    max_len = max(len(r) for r in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (t, m) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(t)] = torch.tensor(t)
        mask[i, :len(m)] = torch.tensor(m)
    return tokens, mask, label

def iterate_examples(split):
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")) as f:
        for line in f:
            yield json.loads(line)

# --------------------------------------------------------------------
@torch.no_grad()
def evaluate(model_name="rhysjones/gpt2-124M-edu-fineweb-10B"):
    # pick best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # load model/tokenizer from HF
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    torch.set_float32_matmul_precision('high')

    num_correct = num_correct_norm = num_total = 0
    for example in iterate_examples("val"):
        tokens, mask, label = render_example(example)
        tokens, mask = tokens.to(device), mask.to(device)

        logits = model(tokens).logits

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
            print(f"{num_total:5d} examples | acc: {acc:.4f} | acc_norm: {acc_norm:.4f}")

    acc = num_correct / num_total
    acc_norm = num_correct_norm / num_total
    print("\n================ FINAL RESULTS ================")
    print(f"Model: {model_name}")
    print(f"Accuracy (raw):      {acc:.4f}")
    print(f"Accuracy (normalized): {acc_norm:.4f}")
    print("================================================")

# --------------------------------------------------------------------
if __name__ == "__main__":
    evaluate()

# ================ FINAL RESULTS ================
# Model: rhysjones/gpt2-124M-edu-fineweb-10B
# Accuracy (raw):      0.2956
# Accuracy (normalized): 0.3098
# ================================================
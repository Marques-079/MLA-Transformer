import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    device = pick_device()
    print("Using device:", device)

    tok = AutoTokenizer.from_pretrained("gpt2")  # 124M
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval().to(device)

    prompt = "Tell me about the Video game Minecaft and what you do when you play the game"
    inputs = tok(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            use_cache=True,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,  # silence pad/eos warning on gpt2
        )

    print("\n=== Output ===")
    print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()

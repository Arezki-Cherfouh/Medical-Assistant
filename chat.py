import argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# ── Args ──────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--model", default="./merged", help="Path to merged model folder")
p.add_argument("--max_new_tokens", type=int, default=512)
p.add_argument("--temperature", type=float, default=0.3)
p.add_argument("--top_p", type=float, default=0.9)
args = p.parse_args()

SYS = "You are a clinical AI. Reason step-by-step; if urgent, tell the user to seek emergency care."

# ── Load ──────────────────────────────────────────────────────
print(f"\n🔬 Loading model from {args.model} …")
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    low_cpu_mem_usage=True,
)
model.eval()
streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
device = next(model.parameters()).device
print(f"✅ Ready on {device}  (type 'exit' or 'quit' to stop, 'clear' to reset)\n")

# ── Chat loop ─────────────────────────────────────────────────
history = [{"role": "system", "content": SYS}]

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!"); break

    if not user_input:
        continue
    if user_input.lower() in ("exit", "quit"):
        print("Bye!"); break
    if user_input.lower() == "clear":
        history = [{"role": "system", "content": SYS}]
        print("🗑️  History cleared.\n"); continue

    history.append({"role": "user", "content": user_input})

    prompt = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)

    print("\nAssistant: ", end="", flush=True)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tok.eos_token_id,
        )

    # decode only the new tokens for history
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    reply = tok.decode(new_tokens, skip_special_tokens=True).strip()
    history.append({"role": "assistant", "content": reply})
    print()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
from huggingface_hub import login

# ── Config ────────────────────────────────────────────────────
REPO   = "Arezki-Cherfouh/llama-3.1-8b-med-expert"
HF_TOKEN = input("HF Token (or press Enter to skip): ").strip() or None
if HF_TOKEN:
    login(token=HF_TOKEN)

MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.7
TOP_P          = 0.9
REP_PENALTY    = 1.3   # stops repetition loops

SYS = "You are a clinical AI assistant. Answer the user's medical question clearly and concisely. Reason step-by-step. Do not repeat yourself. If the situation is urgent, tell the user to seek emergency care immediately."

# ── Load ──────────────────────────────────────────────────────
print(f"\n🔬 Loading {REPO} ...")

tok = AutoTokenizer.from_pretrained(REPO, token=HF_TOKEN)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    REPO,
    token=HF_TOKEN,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval()

streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
device   = next(model.parameters()).device
print(f"✅ Ready on {device}  (type 'exit' to quit, 'clear' to reset)\n")

# ── Chat loop ─────────────────────────────────────────────────
history = []

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
        history = []
        print("🗑️  History cleared.\n"); continue

    history.append({"role": "user", "content": user_input})

    # Build prompt manually — avoids chat template EOS issue
    prompt = "<|begin_of_text|>"
    prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{SYS}<|eot_id|>"
    for turn in history:
        prompt += f"<|start_header_id|>{turn['role']}<|end_header_id|>\n\n{turn['content']}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    inputs = tok(prompt, return_tensors="pt").to(device)

    print("\nAssistant: ", end="", flush=True)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REP_PENALTY,
            streamer=streamer,
            pad_token_id=128256,
            eos_token_id=128009,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    reply = tok.decode(new_tokens, skip_special_tokens=True).strip()
    history.append({"role": "assistant", "content": reply})
    print()
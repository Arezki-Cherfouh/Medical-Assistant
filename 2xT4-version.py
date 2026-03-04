!pip install -q trl peft accelerate bitsandbytes datasets transformers

from kaggle_secrets import UserSecretsClient
import os

# Pre-verify your token so you don't waste 9 hours on a logic error
try:
    token = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = token
    print("✅ HF_TOKEN detected and loaded.")
except:
    print("❌ ERROR: HF_TOKEN not found in Add-ons > Secrets!")






%%writefile qlora_dual_t4.py

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch, gc
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
from kaggle_secrets import UserSecretsClient

# ── 1. Setup & Auth ──────────────────────────────────────────
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

BASE   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUT    = "/kaggle/working/adapter"
MERGED = "/kaggle/working/merged"

print(f"🚀 Single-process, dual-T4 model-parallel training")
print(f"   Visible GPUs: {torch.cuda.device_count()}")

# ── 2. Tokenizer ─────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained(BASE)
tok.add_special_tokens({"pad_token": "<|pad|>"})
tok.padding_side = "right"

# ── 3. Model Loading ─────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
)

model.resize_token_embeddings(len(tok), pad_to_multiple_of=8, mean_resizing=False)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora_config)

# ── Cast LoRA adapter weights to fp32 manually ───────────────
# prepare_model_for_kbit_training can leave adapters in bf16.
# We cast only the trainable LoRA params (tiny: 0.5% of model)
# to fp32 so the optimizer never touches bf16 tensors.
for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)

model.print_trainable_parameters()

# ── 4. Dataset Helpers ───────────────────────────────────────
SYS = "You are a clinical AI. Reason step-by-step; if urgent, tell the user to seek emergency care."

def to_text(messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def load_stage1(n=6000):
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train").shuffle(42).select(range(n))
    def fmt(x):
        opts = "\n".join(f"{k}) {v}" for k, v in x["options"].items())
        return {"text": to_text([
            {"role": "system",    "content": SYS},
            {"role": "user",      "content": f"{x['question']}\n{opts}"},
            {"role": "assistant", "content": f"The answer is {x['answer_idx']}) {x['options'][x['answer_idx']]}"},
        ])}
    return ds.map(fmt, remove_columns=ds.column_names)

def load_stage2(n=4000):
    ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train").shuffle(42).select(range(n))
    def fmt(x):
        cot = x.get("complex_cot") or x.get("thinking", "")
        ans = x.get("response")    or x.get("output", "")
        return {"text": to_text([
            {"role": "system",    "content": SYS},
            {"role": "user",      "content": x.get("question", "")},
            {"role": "assistant", "content": f"<thinking>\n{cot}\n</thinking>\n\n{ans}"},
        ])}
    return ds.map(fmt, remove_columns=ds.column_names)

def load_stage3(n=1000):
    ds = load_dataset("Mohammed-Altaf/medical-instruction-100k", split="train").shuffle(42).select(range(n))
    TRIAGE = ["chest pain","can't breathe","unconscious","stroke",
              "severe bleeding","overdose","seizure"]
    def fmt(x):
        q   = (x.get("input") or x.get("instruction", "")).lower()
        ans = x.get("output", "")
        if any(t in q for t in TRIAGE):
            ans = "⚠️ EMERGENCY: Please call 911 or go to the ER immediately.\n\n" + ans
        return {"text": to_text([
            {"role": "system",    "content": SYS},
            {"role": "user",      "content": x.get("input") or x.get("instruction", "")},
            {"role": "assistant", "content": ans},
        ])}
    return ds.map(fmt, remove_columns=ds.column_names)

# ── 5. Training Function ─────────────────────────────────────
def train_stage(dataset, stage_name, epochs=1, lr=2e-4):
    print(f"\n{'='*55}\n  {stage_name}\n{'='*55}")
    split = dataset.train_test_split(test_size=0.02, seed=42)

    config = SFTConfig(
        output_dir=OUT,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        max_grad_norm=0.3,
        # ── No AMP scaler: avoids bf16/fp16 conflict ──────
        # 4-bit compute still uses fp16 internally via bnb.
        bf16=False,
        fp16=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        report_to="none",
        dataset_text_field="text",
        max_length=1024,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tok,
    )
    trainer.train()

# ── 6. Execute Stages ────────────────────────────────────────
train_stage(load_stage1(), "Stage 1 — MedQA USMLE",          epochs=1, lr=2e-4)
train_stage(load_stage2(), "Stage 2 — Medical-o1 Reasoning",  epochs=1, lr=1e-4)
train_stage(load_stage3(), "Stage 3 — Medical Instructions",  epochs=1, lr=5e-5)

# ── 7. Save adapter ──────────────────────────────────────────
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print(f"\n✅ Adapter saved to {OUT}")

# ── 8. Merge ─────────────────────────────────────────────────
del model
gc.collect()
torch.cuda.empty_cache()

print("\nMerging adapter into base model (fp16)…")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
merged_model = PeftModel.from_pretrained(base_model, OUT)
merged_model = merged_model.merge_and_unload()

# merged_model.save_pretrained(MERGED, safe_serialization=True, max_shard_size="4GB")
# tok.save_pretrained(MERGED)

# ── 9. Push to Hub ───────────────────────────────────────────
repo_id = "Arezki-Cherfouf/llama-3.1-8b-med-expert"
merged_model.push_to_hub(repo_id, private=False)
tok.push_to_hub(repo_id, private=True)
print(f"\n✅ SUCCESS: Model pushed to {repo_id}")






!PYTORCH_ALLOC_CONF=expandable_segments:True python qlora_dual_t4.py
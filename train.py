# ================================================================
# Multi-Stage QLoRA Fine-Tune: LLaMA 3.1 8B → Medical Expert
# ================================================================

import torch, os, gc, sys
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
from kaggle_secrets import UserSecretsClient

# ── 1. Setup & Auth ──────────────────────────────────────────
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
BASE   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUT    = "/kaggle/working/adapter"
MERGED = "/kaggle/working/merged"

# ── 2. Tokenizer ─────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained(BASE)
tok.add_special_tokens({"pad_token": "<|pad|>"})
tok.padding_side = "right"

# ── 3. Optimized Model Loading (OOM Fix) ─────────────────────
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True
    ),
    device_map="auto", 
    dtype=torch.bfloat16,        # Updated from torch_dtype
    low_cpu_mem_usage=True,      # Prevents CPU/RAM spike
    attn_implementation="sdpa"    # Efficient attention
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
# model.resize_token_embeddings(len(tok))
model.resize_token_embeddings(len(tok), mean_resizing=False)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
))

# ── 4. Dataset Helpers ───────────────────────────────────────
SYS = "You are a clinical AI. Reason step-by-step; if urgent, tell the user to seek emergency care."

def to_text(messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def load_stage1(n=6000):
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train").shuffle(42).select(range(n))
    def fmt(x):
        opts = "\n".join(f"{k}) {v}" for k,v in x["options"].items())
        return {"text": to_text([
            {"role":"system",    "content": SYS},
            {"role":"user",      "content": f"{x['question']}\n{opts}"},
            {"role":"assistant", "content": f"The answer is {x['answer_idx']}) {x['options'][x['answer_idx']]}"},
        ])}
    return ds.map(fmt, remove_columns=ds.column_names)

def load_stage2(n=4000):
    ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train").shuffle(42).select(range(n))
    def fmt(x):
        cot = x.get("complex_cot") or x.get("thinking","")
        ans = x.get("response")    or x.get("output","")
        return {"text": to_text([
            {"role":"system",    "content": SYS},
            {"role":"user",      "content": x.get("question","")},
            {"role":"assistant", "content": f"<thinking>\n{cot}\n</thinking>\n\n{ans}"},
        ])}
    return ds.map(fmt, remove_columns=ds.column_names)

def load_stage3(n=1000):
    ds = load_dataset("Mohammed-Altaf/medical-instruction-100k", split="train").shuffle(42).select(range(n))
    TRIAGE = ["chest pain","can't breathe","unconscious","stroke","severe bleeding","overdose","seizure"]
    def fmt(x):
        q = (x.get("input") or x.get("instruction","")).lower()
        ans = x.get("output","")
        if any(t in q for t in TRIAGE):
            ans = "⚠️ EMERGENCY: Please call 911 or go to the ER immediately.\n\n" + ans
        return {"text": to_text([
            {"role":"system",    "content": SYS},
            {"role":"user",      "content": x.get("input") or x.get("instruction","")},
            {"role":"assistant", "content": ans},
        ])}
    return ds.map(fmt, remove_columns=ds.column_names)

# ── 5. Training Function (TRL 0.12+ Fix) ────────────────────
def train_stage(dataset, stage_name, epochs=1, lr=2e-4):
    print(f"\n{'='*50}\n  {stage_name}\n{'='*50}")
    split = dataset.train_test_split(test_size=0.02, seed=42)
    
    # In latest TRL, sft-specific args are bundled in SFTConfig
    config = SFTConfig(
        output_dir=OUT,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="no",
        report_to="none",
        max_grad_norm=0.3,
        dataset_text_field="text",
        max_length=1536,             # Reduced to fit VRAM (renamed from max_seq_length in TRL 0.13+)
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=split["train"], 
        eval_dataset=split["test"],
        processing_class=tok,        # New parameter name for tokenizer
    )
    trainer.train()

# ── 6. Execute Stages ────────────────────────────────────────
train_stage(load_stage1(), "Stage 1: MedQA",      epochs=1, lr=2e-4)
train_stage(load_stage2(), "Stage 2: Medical-o1", epochs=1, lr=1e-4)
train_stage(load_stage3(), "Stage 3: MedInst",    epochs=1, lr=5e-5)

# ── 7. Save Adapter & Clean VRAM ─────────────────────────────
model.save_pretrained(OUT)
tok.save_pretrained(OUT)

del model
gc.collect()
torch.cuda.empty_cache() # Crucial for freeing memory for the merge

# ── 8. Merge in FP16 (Avoids 4-bit loading issues) ───────────
print("\nMerging model components...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE, 
    dtype=torch.float16,     # Use FP16 for the merge
    device_map="auto",
    low_cpu_mem_usage=True
)
merged_model = PeftModel.from_pretrained(base_model, OUT)
merged_model = merged_model.merge_and_unload()

# Save final version
merged_model.save_pretrained(MERGED, safe_serialization=True, max_shard_size="4GB")
tok.save_pretrained(MERGED)

# ── 9. Push to Hub ───────────────────────────────────────────
repo_id = "Arezki-Cherfouh/llama-3.1-8b-med-expert"
merged_model.push_to_hub(repo_id, private=True)
tok.push_to_hub(repo_id, private=True)

print(f"\n✅ SUCCESS: Model pushed to {repo_id}")
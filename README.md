# 🏥 LLaMA 3.1 8B Medical — Multi-Stage QLoRA Fine-Tune

Fine-tunes **Meta LLaMA 3.1 8B Instruct** into a clinical reasoning assistant using a 3-stage QLoRA pipeline, then lets you chat with it locally.

## Stages

| #   | Dataset                  | Purpose                     | LR   |
| --- | ------------------------ | --------------------------- | ---- |
| 1   | MedQA USMLE              | Factual knowledge anchoring | 2e-4 |
| 2   | Medical-o1 Reasoning SFT | Chain-of-thought reasoning  | 1e-4 |
| 3   | MedInstruct (subset)     | Safety & triage alignment   | 5e-5 |

## Files

```
train.ipynb   # Single-cell Kaggle notebook (runs end-to-end)
chat.py       # Local CLI chat with streaming
requirements.txt
```

## Usage

### Train (Kaggle)

1. Enable **GPU T4 x2** and **Internet** in Kaggle settings
2. Add your Hugging Face token (needs Llama 3.1 access)
3. Run the single cell — datasets and model download automatically
4. Download `/kaggle/working/merged` when done

### Chat (local)

```bash
pip install -r requirements.txt
python chat.py --model ./merged
```

| Command         | Action                     |
| --------------- | -------------------------- |
| `clear`         | Reset conversation history |
| `exit` / `quit` | Quit                       |

## Requirements

- **Kaggle training:** T4 x2 GPU (~3–4 hrs), HF account with Llama 3.1 access
- **Local inference:** NVIDIA GPU (8 GB+ VRAM) recommended; CPU works but is slow

## License

Base model weights: [Meta Llama 3.1 Community License](https://llama.meta.com/llama3_1/license/).  
Training code: MIT.

"""Classifica reviews com o adapter publicado no Hugging Face.

Uso:
  uv run --group train python src/predict.py "Chegou rápido e bem embalado" "Veio quebrado"
"""
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = "Qwen/Qwen2.5-0.5B"
ADAPTER = "rmascheti/qwen2.5-0.5b-reviews-ptbr"
LABELS = ["negativo", "neutro", "positivo"]


def load():
    tok = AutoTokenizer.from_pretrained(ADAPTER)
    base = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=len(LABELS))
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, ADAPTER).eval()  # carrega LoRA + cabeça de classificação treinada
    return tok, model


def main() -> None:
    texts = sys.argv[1:] or ["Produto excelente, chegou antes do prazo.", "Bom, mas a entrega atrasou uma semana.", "Veio quebrado e ninguém responde."]
    tok, model = load()
    with torch.no_grad():
        batch = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        probs = model(**batch).logits.softmax(-1)
    for text, p in zip(texts, probs):
        print(f"{LABELS[int(p.argmax())]:>8}  {p.max():.2f}  {text}")


if __name__ == "__main__":
    main()

"""Fine-tuning LoRA de um LLM pequeno (Qwen2.5-0.5B) para classificar sentimento de reviews PT-BR.

Feito para rodar numa GPU T4 gratuita do Kaggle (fp16). Na sua máquina, use --smoke
para validar o pipeline em segundos.

Uso:
  uv run --group train python src/train_lora.py --smoke            # teste rápido (CPU ok)
  python src/train_lora.py --train-size 40000 --push-to-hub USER/qwen2.5-0.5b-reviews-ptbr
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["negativo", "neutro", "positivo"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def load_split(name: str, n: int | None, seed: int) -> Dataset:
    df = pd.read_parquet(ROOT / f"data/{name}.parquet")
    if n and n < len(df):
        # amostra estratificada para manter a proporção de classes
        df = df.groupby("label", group_keys=False).sample(frac=n / len(df), random_state=seed)
    df = df.assign(labels=df["label"].map(LABEL2ID))[["text", "labels"]]
    return Dataset.from_pandas(df, preserve_index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--train-size", type=int, default=40_000)
    p.add_argument("--eval-size", type=int, default=2_000)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(ROOT / "outputs/qwen-lora"))
    p.add_argument("--push-to-hub", default=None, help="repo id no Hugging Face, ex.: usuario/modelo")
    p.add_argument("--smoke", action="store_true", help="roda com poucos exemplos para testar o pipeline")
    args = p.parse_args()

    if args.smoke:
        args.train_size, args.eval_size, args.epochs, args.batch_size = 64, 32, 0.25, 4

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.pad_token or tok.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL2ID,
        dtype=torch.float32,  # pesos em fp32; o Trainer usa autocast fp16 na GPU
    )
    model.config.pad_token_id = tok.pad_token_id
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    train = load_split("train", args.train_size, args.seed).map(tokenize, batched=True, remove_columns=["text"])
    val = load_split("val", args.eval_size, args.seed).map(tokenize, batched=True, remove_columns=["text"])

    def metrics(eval_pred):
        logits, labels = eval_pred
        pred = np.argmax(logits, axis=-1)
        return {"macro_f1": f1_score(labels, pred, average="macro"), "accuracy": (pred == labels).mean()}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            warmup_steps=0.05,  # float = fração do total de passos
            lr_scheduler_type="cosine",
            fp16=torch.cuda.is_available(),
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            logging_steps=50,
            report_to="none",
            seed=args.seed,
        ),
        train_dataset=train,
        eval_dataset=val,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics,
    )
    trainer.train()

    # avaliação final no test completo (mesmo split do baseline)
    test_df = pd.read_parquet(ROOT / "data/test.parquet")
    if args.smoke:
        test_df = test_df.head(32)
    test = Dataset.from_pandas(test_df.assign(labels=test_df["label"].map(LABEL2ID))[["text", "labels"]])
    test = test.map(tokenize, batched=True, remove_columns=["text"])
    t0 = time.perf_counter()
    out = trainer.predict(test)
    ms_per_review = (time.perf_counter() - t0) / len(test) * 1000
    pred = out.predictions.argmax(-1)

    print(classification_report(out.label_ids, pred, target_names=LABELS, digits=3))
    result = {
        "model": f"{args.model} + LoRA r={args.lora_r}",
        "macro_f1": round(float(f1_score(out.label_ids, pred, average="macro")), 4),
        "accuracy": round(float((pred == out.label_ids).mean()), 4),
        "train_size": len(train),
        "ms_per_review": round(ms_per_review, 3),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "n_test": len(test),
    }
    print(result)
    (ROOT / "reports").mkdir(exist_ok=True)
    name = "lora_smoke.json" if args.smoke else "lora.json"
    (ROOT / "reports" / name).write_text(json.dumps(result, indent=2))

    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    if args.push_to_hub and not args.smoke:
        trainer.model.push_to_hub(args.push_to_hub)
        tok.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()

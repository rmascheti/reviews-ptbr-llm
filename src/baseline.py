"""Baseline clássico: TF-IDF + Regressão Logística.

É a régua que o LLM fine-tunado precisa superar para justificar seu custo.
Uso: uv run python src/baseline.py
"""
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    train = pd.read_parquet(ROOT / "data/train.parquet")
    test = pd.read_parquet(ROOT / "data/test.parquet")

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"),
    )
    t0 = time.perf_counter()
    model.fit(train["text"], train["label"])
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = model.predict(test["text"])
    ms_per_review = (time.perf_counter() - t0) / len(test) * 1000

    print(classification_report(test["label"], pred, digits=3))
    result = {
        "model": "tfidf-logreg",
        "macro_f1": round(f1_score(test["label"], pred, average="macro"), 4),
        "accuracy": round(float((pred == test["label"]).mean()), 4),
        "train_seconds": round(train_s, 1),
        "ms_per_review_cpu": round(ms_per_review, 4),
        "n_test": len(test),
    }
    print(result)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/baseline.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

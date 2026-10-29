"""Limpa o B2W-Reviews01 e gera splits estratificados train/val/test.

Sentimento derivado da nota: 1-2 = negativo, 3 = neutro, 4-5 = positivo.
Uso: uv run python src/prepare_data.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parents[1] / "data"
SEED = 42


def sentiment(rating: int) -> str:
    if rating <= 2:
        return "negativo"
    if rating == 3:
        return "neutro"
    return "positivo"


def main() -> None:
    df = pd.read_csv(DATA / "B2W-Reviews01.csv", low_memory=False)
    df = df[["review_title", "review_text", "overall_rating", "site_category_lv1"]]
    df = df.dropna(subset=["review_text", "overall_rating"])
    df["review_title"] = df["review_title"].fillna("")
    df["text"] = (df["review_title"].str.strip() + ". " + df["review_text"].str.strip()).str.strip(". ")
    df = df[df["text"].str.len().between(20, 1500)].drop_duplicates(subset="text")
    df["label"] = df["overall_rating"].astype(int).map(sentiment)

    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=SEED)

    cols = ["text", "label", "overall_rating", "site_category_lv1"]
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part[cols].to_parquet(DATA / f"{name}.parquet", index=False)
        print(f"{name}: {len(part):>6}  {part['label'].value_counts(normalize=True).round(3).to_dict()}")


if __name__ == "__main__":
    main()

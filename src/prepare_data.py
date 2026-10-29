"""Limpa o B2W-Reviews01 e gera splits estratificados train/val/test.

Sentimento derivado da nota: 1-2 = negativo, 3 = neutro, 4-5 = positivo.
Uso: uv run python src/prepare_data.py
"""
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parents[1] / "data"
CSV_URL = "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/main/B2W-Reviews01.csv"
SEED = 42


def download_if_missing() -> Path:
    path = DATA / "B2W-Reviews01.csv"
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)  # data/ não vem no clone: o git não versiona pasta vazia
        print(f"baixando {CSV_URL}")
        urllib.request.urlretrieve(CSV_URL, path)
    return path


def sentiment(rating: int) -> str:
    if rating <= 2:
        return "negativo"
    if rating == 3:
        return "neutro"
    return "positivo"


def main() -> None:
    df = pd.read_csv(download_if_missing(), low_memory=False)
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

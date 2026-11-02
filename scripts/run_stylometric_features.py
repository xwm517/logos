import pandas as pd

from logos.paths import DATA_DIR
from logos.features.stylometric import FEATURE_NAMES, ensure_nltk_data, features_frame

MAGE_DIR = DATA_DIR / "mage"
OUT_PATH = MAGE_DIR / "features" / "stylometric.parquet"


def main():
    ensure_nltk_data()
    corpus = pd.read_parquet(MAGE_DIR / "corpus.parquet", columns=["text_id", "text"])

    print(f"computing features for {len(corpus):,} texts", flush=True)

    features = features_frame(corpus["text"].tolist(), show_progress=True)
    features.insert(0, "text_id", corpus["text_id"].to_numpy())

    blank = (features[FEATURE_NAMES] == 0).all(axis=1).sum()
    print(f"{blank} texts with no usable words")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUT_PATH, index=False)

    print(f"saved {len(features):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()

import pandas as pd
from tqdm import tqdm

from logos.paths import DATA_DIR
from logos.detectors.binoculars import Binoculars

MAGE_DIR = DATA_DIR / "mage"
OUTPUT_PATH = MAGE_DIR / "scores" / "binoculars" / "tune.parquet"

BATCH_SIZE = 8


def main():
    corpus = pd.read_parquet(MAGE_DIR / "corpus.parquet", columns=["text_id", "text"])
    membership = pd.read_parquet(
        MAGE_DIR / "training_sets.parquet", columns=["text_id", "part"]
    )

    tune_ids = membership.loc[membership["part"] == "tune", "text_id"].unique()
    texts = corpus[corpus["text_id"].isin(tune_ids)]

    print(f"scoring {len(texts)} texts")

    detector = Binoculars()

    scores = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts["text"].iloc[start:start + BATCH_SIZE].tolist()
        scores += detector.compute_score(batch)

    output = pd.DataFrame({
        "text_id": texts["text_id"].to_numpy(),
        "part": "tune",
        "score": [-s for s in scores],
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)

    print(f"saved {len(output)} scores to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

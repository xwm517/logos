import pandas as pd
from tqdm import tqdm

from logos.paths import DATA_DIR
from logos.evaluation.folds import calibration_set, test_set
from logos.detectors.binoculars import Binoculars

MAGE_DIR = DATA_DIR / "mage"
OUT_PATH = MAGE_DIR / "scores" / "binoculars" / "all.parquet"

# The cross-perplexity step builds a probability tensor over Falcon's full
# 65,024-token vocabulary for every position in the batch, which is about a
# gigabyte at this size on top of the 28 GB of model weights
BATCH_SIZE = 8


def main():
    mage_df = pd.read_parquet(MAGE_DIR / "corpus.parquet")

    texts = pd.concat([
        calibration_set(mage_df).assign(part="calibration"),
        test_set(mage_df).assign(part="test"),
    ])

    print(f"scoring {len(texts)} texts")

    detector = Binoculars()

    scores = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts["text"].iloc[start:start + BATCH_SIZE].tolist()
        scores += detector.compute_score(batch)

    # Flip the sign so that, as with every other detector in this study, a
    # higher score means more likely machine-generated
    out = pd.DataFrame({
        "text_id": texts["text_id"].to_numpy(),
        "part": texts["part"].to_numpy(),
        "score": [-s for s in scores],
    })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    print(f"saved {len(out)} scores to {OUT_PATH}")


if __name__ == "__main__":
    main()
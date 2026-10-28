import pandas as pd
from tqdm import tqdm

from logos.paths import DATA_DIR
from logos.evaluation.folds import calibration_set, test_set
from logos.models.binoculars import Binoculars

MAGE_DIR = DATA_DIR / "mage"
OUT_PATH = MAGE_DIR / "scores" / "binoculars" / "all.parquet"
BATCH_SIZE = 16


def main():
    # Nothing to train, so score the calibration humans and the whole test split once
    corpus = pd.read_parquet(MAGE_DIR / "corpus.parquet")
    texts = pd.concat([
        calibration_set(corpus).assign(part="calibration"),
        test_set(corpus).assign(part="test"),
    ])

    detector = Binoculars()
    scores = []
    for start in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts["text"].iloc[start:start + BATCH_SIZE].tolist()
        scores += detector.score(batch)

    # Flip the sign so that, like every other detector, higher means more likely machine
    out = pd.DataFrame({"text_id": texts["text_id"].to_numpy(), "part": texts["part"].to_numpy(),
                        "score": [-s for s in scores]})
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print("scored", len(out), "texts; saved to", OUT_PATH)


if __name__ == "__main__":
    main()
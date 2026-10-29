import numpy as np
import pandas as pd
from tqdm import tqdm

from logos.paths import DATA_DIR
from logos.project_data.hc3 import load_hc3, clean_hc3
from logos.project_data.preprocess import preprocess, train_val_split
from logos.evaluation.metrics import (
    calibrate_threshold, tpr_at_threshold, fpr_at_threshold, auroc
)
from logos.detectors.binoculars import Binoculars

BATCH_SIZE = 8


def main():
    hc3 = preprocess(clean_hc3(load_hc3()))
    _, val_df = train_val_split(hc3, val_fraction=0.15)

    labels = val_df["label"].to_numpy()
    print(f"HC3 validation: {len(val_df)} texts, "
          f"{(labels == 0).sum()} human, {(labels == 1).sum()} ChatGPT")
    print("(notebook 02 reports 11,708 texts, 7,785 human and 3,923 ChatGPT; if "
          "these differ, the split is not the one the lexical baseline used)")

    detector = Binoculars()

    texts = val_df["text"].tolist()
    raw = []
    for start in tqdm(range(0, len(texts), BATCH_SIZE)):
        raw += detector.compute_score(texts[start:start + BATCH_SIZE])

    scores = np.array([-score for score in raw])

    tau = calibrate_threshold(scores, labels, target_fpr=0.01)

    print("\n--- Binoculars on HC3 validation ---")
    print(f"tau*       {tau:.6f}")
    print(f"FPR        {fpr_at_threshold(scores, labels, tau):.4f}")
    print(f"TPR@1%FPR  {tpr_at_threshold(scores, labels, tau):.4f}")
    print(f"AUROC      {auroc(scores, labels):.4f}")

    print(f"\nmean score, human texts   {scores[labels == 0].mean():.4f}")
    print(f"mean score, ChatGPT texts {scores[labels == 1].mean():.4f}")

    print("\nfor comparison")
    print("  lexical baseline on HC3, tuned C   TPR 0.791")
    print("  lexical baseline on MAGE, LOFO     TPR 0.197")
    print("  Binoculars on MAGE, mean           TPR 0.109")

    by_domain = val_df.assign(flagged=scores >= tau)
    humans = by_domain[by_domain["label"] == 0]
    chatgpt = by_domain[by_domain["label"] == 1]
    print("\nby domain")
    print(pd.DataFrame({
        "n_human": humans.groupby("domain").size(),
        "FPR": humans.groupby("domain")["flagged"].mean(),
        "TPR": chatgpt.groupby("domain")["flagged"].mean(),
    }).round(4))

    out_path = DATA_DIR / "hc3" / "scores" / "binoculars" / "validation.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    val_df.assign(score=scores)[["label", "domain", "score"]].to_parquet(
        out_path, index=False
    )
    print(f"\nsaved scores to {out_path}")


if __name__ == "__main__":
    main()
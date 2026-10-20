import argparse

import pandas as pd

from logos.paths import DATA_DIR
from logos.project_data.mage import families
from logos.evaluation.folds import load_training_set, calibration_set, test_set
from logos.models.tf_idf import fit_lexical_baseline

MAGE_DIR = DATA_DIR / "mage"
SCORES_DIR = MAGE_DIR / "scores" / "lexical"
SEEDS = [0, 1, 2, 3, 4]
SET_NAMES = ["In-distribution"] + [f"LOFO {family}" for family in families]


def main():
    parser = argparse.ArgumentParser(description="Fit the lexical baseline on one training set and save its scores")
    parser.add_argument("--task", type=int, required=True, help="0 to 39")
    args = parser.parse_args()

    seed = SEEDS[args.task // len(SET_NAMES)]
    set_name = SET_NAMES[args.task % len(SET_NAMES)]

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCORES_DIR / f"seed{seed}_{set_name.replace(' ', '_')}.parquet"
    if out_path.exists():
        print(f"seed {seed}, {set_name}: already saved, skipping")
        return

    corpus = pd.read_parquet(MAGE_DIR / "corpus.parquet")
    membership = pd.read_parquet(MAGE_DIR / "training_sets.parquet")
    calibration = calibration_set(corpus)
    test = test_set(corpus)

    train_df = load_training_set(corpus, membership, seed=seed, name=set_name)
    train_part = train_df[train_df["part"] == "train"]
    tune_part = train_df[train_df["part"] == "tune"]
    model, best_c, accuracies = fit_lexical_baseline(
        train_part["text"], train_part["label"], tune_part["text"], tune_part["label"], seed=seed)

    scores = pd.concat([
        pd.DataFrame({"text_id": tune_part["text_id"], "part": "tune",
                      "score": model.predict_proba(tune_part["text"])[:, 1]}),
        pd.DataFrame({"text_id": calibration["text_id"], "part": "calibration",
                      "score": model.predict_proba(calibration["text"])[:, 1]}),
        pd.DataFrame({"text_id": test["text_id"], "part": "test",
                      "score": model.predict_proba(test["text"])[:, 1]}),
    ])

    scores.assign(seed=seed, training_set=set_name, c=best_c).to_parquet(out_path, index=False)
    print(f"seed {seed}, {set_name}: C = {best_c:g}, tune accuracy {accuracies[best_c]:.4f}")


if __name__ == "__main__":
    main()
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast

from logos.paths import DATA_DIR
from logos.project_data.mage import families
from logos.evaluation.folds import load_training_set, calibration_set, test_set
from logos.features.stylometric import FEATURE_NAMES
from logos.detectors.stylometric import StylometricDataset, StylometricDetector, score, train

MAGE_DIR = DATA_DIR / "mage"
FEATURES_PATH = MAGE_DIR / "features" / "stylometric.parquet"
SCORES_DIR = MAGE_DIR / "scores" / "stylometric"
MODELS_DIR = MAGE_DIR / "models" / "stylometric"

SEEDS = [0, 1, 2, 3, 4]
SET_NAMES = ["In-distribution"] + [f"LOFO {family}" for family in families]

CHECKPOINT = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5


def feature_rows(df, features):
    return features.loc[df["text_id"], FEATURE_NAMES].to_numpy()


def loader_for(df, features, tokenizer, shuffle=False):
    data = StylometricDataset(
        texts=df["text"],
        labels=df["label"],
        features=feature_rows(df, features),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
    return DataLoader(data, batch_size=BATCH_SIZE, shuffle=shuffle)


def main():
    parser = argparse.ArgumentParser(
        description="Train the stylometric-neural detector on one training set and save its scores")
    parser.add_argument("--task", type=int, required=True, help="0 to 39")
    args = parser.parse_args()

    seed = SEEDS[args.task // len(SET_NAMES)]
    set_name = SET_NAMES[args.task % len(SET_NAMES)]

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCORES_DIR / f"seed{seed}_{set_name.replace(' ', '_')}.parquet"
    if out_path.exists():
        print(f"seed {seed}, {set_name}: already saved, skipping")
        return

    if not FEATURES_PATH.exists():
        raise SystemExit(f"{FEATURES_PATH} is missing. Run run_stylometric_features.py first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"seed {seed}, {set_name}, on {device}", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    corpus = pd.read_parquet(MAGE_DIR / "corpus.parquet")
    membership = pd.read_parquet(MAGE_DIR / "training_sets.parquet")
    features = pd.read_parquet(FEATURES_PATH).set_index("text_id")

    calibration = calibration_set(corpus)
    test = test_set(corpus)

    train_df = load_training_set(corpus, membership, seed=seed, name=set_name)
    train_part = train_df[train_df["part"] == "train"]
    tune_part = train_df[train_df["part"] == "tune"]

    scaler = StandardScaler().fit(feature_rows(train_part, features))
    scaled = features.copy()
    scaled[FEATURE_NAMES] = scaler.transform(features[FEATURE_NAMES])

    tokenizer = DistilBertTokenizerFast.from_pretrained(CHECKPOINT)

    print(f"training on {len(train_part):,} texts, tuning slice {len(tune_part):,}", flush=True)

    model = StylometricDetector()
    train(
        model,
        loader_for(train_part, scaled, tokenizer, shuffle=True),
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        device=device,
    )

    parts = {"tune": tune_part, "calibration": calibration, "test": test}
    scores = []

    for part_name, part_df in parts.items():
        print(f"scoring {part_name}: {len(part_df):,} texts", flush=True)
        scores.append(pd.DataFrame({
            "text_id": part_df["text_id"].to_numpy(),
            "part": part_name,
            "score": score(model, loader_for(part_df, scaled, tokenizer), device=device),
        }))

    out = pd.concat(scores).assign(seed=seed, training_set=set_name)
    out.to_parquet(out_path, index=False)
    print(f"saved {len(out):,} scores to {out_path}")

    if seed == 0:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"seed{seed}_{set_name.replace(' ', '_')}.pt"
        torch.save({"state_dict": model.state_dict(), "scaler": scaler}, model_path)
        print(f"saved the model to {model_path}")


if __name__ == "__main__":
    main()

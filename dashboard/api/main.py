import json
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from logos.paths import DATA_DIR
from logos.project_data.mage import families
from logos.evaluation.folds import calibration_set, test_set
from logos.evaluation.metrics import calibrate_threshold, tpr_at_threshold, fpr_at_threshold, auroc

MAGE_DIR = DATA_DIR / "mage"
SCORES_DIR = MAGE_DIR / "scores"
LABELS = {0: "human", 1: "machine"}

app = FastAPI(title="LOGOS dashboard")

app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["GET"], allow_headers=["*"])


def to_json(frame):
    return json.loads(frame.to_json(orient="index"))


@lru_cache
def corpus():
    return pd.read_parquet(MAGE_DIR / "corpus.parquet",
                           columns=["text_id", "split", "label", "domain", "family", "model", "n_words"])


@lru_cache
def membership():
    return pd.read_parquet(MAGE_DIR / "training_sets.parquet")


@app.get("/api/corpus")
def corpus_summary():
    df = corpus()
    label = df["label"].map(LABELS)
    return {
        "texts": len(df),
        "by_split": to_json(pd.crosstab(df["split"], label)),
        "by_domain": to_json(pd.crosstab(df["domain"], label)),
        "by_family": json.loads(df.loc[df["label"] == 1, "family"].value_counts().to_json()),
        "median_words": to_json(df.groupby(["domain", label])["n_words"].median().unstack()),
    }


@app.get("/api/folds")
def folds_summary():
    m = membership()
    pairs = m.groupby(["training_set", "seed"])["pair_id"].nunique().unstack()
    tune = m[m["part"] == "tune"].groupby(["training_set", "seed"])["pair_id"].nunique().unstack()
    return {"pairs": to_json(pairs), "tune_pairs": to_json(tune)}


@app.get("/api/test-sets")
def test_sets():
    df = corpus()
    rows = []
    for family in families:
        target = test_set(df, target=family)
        rows.append({"family": family,
                     "human": int((target["label"] == 0).sum()),
                     "machine": int((target["label"] == 1).sum())})
    return {"calibration_humans": len(calibration_set(df)), "test": rows}


@app.get("/api/detectors")
def detectors():
    if not SCORES_DIR.exists():
        return []
    return sorted(path.name for path in SCORES_DIR.iterdir() if path.is_dir())


@lru_cache
def results(detector):
    runs = pd.concat([pd.read_parquet(path) for path in sorted((SCORES_DIR / detector).glob("*.parquet"))])
    runs = runs.merge(corpus()[["text_id", "label", "family"]], on="text_id")

    rows = []
    for (seed, set_name), run in runs.groupby(["seed", "training_set"]):
        calibration_part = run[run["part"] == "calibration"]
        test_part = run[run["part"] == "test"]
        tau = calibrate_threshold(calibration_part["score"], calibration_part["label"])
        test_fpr = fpr_at_threshold(test_part["score"], test_part["label"], tau)

        if set_name == "In-distribution":
            condition, targets = "In-distribution", families
        else:
            condition, targets = "LOFO", [set_name.removeprefix("LOFO ")]

        for family in targets:
            family_test = test_part[(test_part["label"] == 0) | (test_part["family"] == family)]
            rows.append({"seed": int(seed), "condition": condition, "family": family,
                         "tpr": tpr_at_threshold(family_test["score"], family_test["label"], tau),
                         "auroc": auroc(family_test["score"], family_test["label"]),
                         "test_fpr": test_fpr})
    return pd.DataFrame(rows)


@app.get("/api/results/{detector}")
def detector_results(detector: str):
    if detector not in detectors():
        raise HTTPException(status_code=404, detail=f"No saved scores for '{detector}'")
    table = results(detector)
    by_family = table.groupby(["family", "condition"])[["tpr", "auroc"]].agg(["mean", "std"])
    by_family.columns = [f"{metric}_{stat}" for metric, stat in by_family.columns]
    per_seed = table.groupby(["condition", "seed"])["tpr"].agg(mean_tpr="mean", worst_tpr="min")
    return {
        "runs": json.loads(table.to_json(orient="records")),
        "by_family": json.loads(by_family.reset_index().to_json(orient="records")),
        "summary": to_json(per_seed.groupby("condition").agg(["mean", "std"]).round(3)),
    }
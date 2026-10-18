import numpy as np

from logos.project_data.mage import families
from logos.project_data.preprocess import length_match_by_domain


def keep_pairs(matched, n_pairs, rng):
    # Pick whole pairs at random, so every human text keeps its machine partner
    chosen = rng.choice(matched["pair_id"].unique(), size=n_pairs, replace=False)
    return matched[matched["pair_id"].isin(chosen)].reset_index(drop=True)


def mark_tuning(matched, rng, share=0.1):
    # Hold back a tenth of the pairs for tuning; detectors never train on these
    pair_ids = matched["pair_id"].unique()
    tuning = rng.choice(pair_ids, size=round(share * len(pair_ids)), replace=False)
    part = np.where(matched["pair_id"].isin(tuning), "tune", "train")
    return matched.assign(part=part)


def training_sets(corpus, seed=0):
    train = corpus[corpus["split"] == "train"]
    humans = train["label"] == 0
    rng = np.random.default_rng(seed)
    sets = {}

    # Leave one family out
    for family in families:
        keep = humans | (train["family"] != family)
        sets[f"LOFO {family}"] = length_match_by_domain(train[keep], seed=seed)

    # Every family in, cut down to the smallest LOFO set so it never has more data to learn from
    smallest = min(lofo["pair_id"].nunique() for lofo in sets.values())
    in_distribution = keep_pairs(length_match_by_domain(train, seed=seed), smallest, rng)
    sets = {"In-distribution": in_distribution, **sets}

    # Hold back the tuning slice in every set
    return {name: mark_tuning(set_df, rng) for name, set_df in sets.items()}


def load_training_set(corpus, membership, seed, name):
    # One saved training set from 04, joined back to its texts
    rows = membership[(membership["seed"] == seed) & (membership["training_set"] == name)]
    return rows.merge(corpus, on="text_id")


def calibration_set(corpus):
    # Human texts from the validation split, only used to set the threshold
    return corpus[(corpus["split"] == "validation") & (corpus["label"] == 0)]


def test_set(corpus, target=None):
    # Test-split humans plus the target family's machine texts.
    # No target means the whole test split, for the in-distribution condition.
    test = corpus[corpus["split"] == "test"]
    if target is None:
        return test
    return test[(test["label"] == 0) | (test["family"] == target)]
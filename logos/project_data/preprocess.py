from sklearn.model_selection import train_test_split


def preprocess(df, min_tokens=15):
    """
    Deduplicate on text and drop fragments shorter than `min_tokens`.
    Applied identically to both corpora so reported rates stay comparable (§3.2).

    The class balance is not changed here. The metric is not affected by imbalance
    because it uses only the human score distribution and only the machine one,
    and class_weight='balanced' keeps all the project_data.
    """
    df = df.drop_duplicates(subset="text")
    n_tokens = df["text"].str.split().map(len)
    return df[n_tokens >= min_tokens].reset_index(drop=True)


def train_val_split(df, val_fraction=0.15, seed=0):
    """
    Use validation only to calibrate the frozen decision threshold tau*; do not train it.
    Stratify the development corpus by label so the human proportion is represented in both parts.
    """
    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        stratify=df["label"],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
from sklearn.model_selection import train_test_split
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re

detokenizer = TreebankWordDetokenizer()

def mark_quotes(text):
    if not re.search(r'\s"\s', text) or text.count('"') % 2:
        return text
    parts = text.split('"')
    marked = parts[0]
    for i, part in enumerate(parts[1:]):
        marked += (" `` " if i % 2 == 0 else " '' ") + part
    return marked


def normalise_text(text, urls=True):
    """
    Before 2023, many NLP datasets were pre-tokenised, so "don't" became "do n't"
    and punctuation was added around it. HC3 for example used older datasets in its human side,
    and its ChatGPT side used fresh text. Leaving the text un-tokenised will cause
    model bias towards these signals in the human text.

    :param text:
    :param urls:
    :return:
    """
    if urls:
        text = re.sub(r"\[([^]]*)]\(\s*(?:URL_\d+|https?://[^\s\"')\]]+)\s*\)", r"\1", text)
        text = re.sub(r"URL_\d+|https?://[^\s\"')\]]+|www\.[^\s\"')\]]+", "", text)
    text = re.sub(r"(?<!\w)\*+|\*+(?!\w)", "", text)
    text = mark_quotes(text)
    return detokenizer.detokenize(text.split())


def length_match(df, tolerance=0.2):
    n = df["text"].str.split().str.len()
    keep = []
    for _, q in df.groupby("question_id"):
        humans = [(i, n[i]) for i in q.index[q["label"] == 0]]
        for m in q.index[q["label"] == 1]:
            candidates = [(abs(n[m] - nh), i) for i, nh in humans if abs(n[m] - nh) <= tolerance * nh]
            if candidates:
                _, h = min(candidates)
                humans = [(i, nh) for i, nh in humans if i != h]
                keep += [h, m]
    return df.loc[keep].reset_index(drop=True)


def preprocess(df, min_tokens=10):
    """
    Deduplicate on text and drop fragments shorter than `min_tokens`.
    Applied identically to both corpora so reported rates stay comparable (§3.2).

    The class balance is not changed here. The metric is not affected by imbalance
    because it uses only the human score distribution and only the machine one,
    and class_weight='balanced' keeps all the project_data.
    """
    df = df.assign(text=df["text"].apply(normalise_text))
    df = df.drop_duplicates(subset="text")
    n_tokens = df["text"].str.split().str.len()
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
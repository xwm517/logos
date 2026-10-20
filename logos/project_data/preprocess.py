import math
import re

import numpy as np
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.model_selection import train_test_split

detokenizer = TreebankWordDetokenizer()


def mark_quotes(text):
    # Only touch texts with a spaced-out quote mark and an even number of quotes,
    # then mark them as opening and closing so the detokeniser can reattach them

    if not re.search(r'\s"\s', text) or text.count('"') % 2:
        return text

    parts = text.split('"')
    marked = parts[0]

    for i, part in enumerate(parts[1:]):
        marked += (" `` " if i % 2 == 0 else " '' ") + part

    return marked


def normalise_text(text, urls=True):
    # Older datasets split words up ("do n't") and spaced out punctuation. HC3's human answers
    # came from those, its ChatGPT answers didn't, so a detector could tell them apart by
    # formatting alone. This puts every text back into normal written form.

    if urls:
        text = re.sub(r"\[([^]]*)]\(\s*(?:URL_\d+|https?://[^\s\"')\]]+)\s*\)", r"\1", text)
        text = re.sub(r"URL_\d+|https?://[^\s\"')\]]+|www\.[^\s\"')\]]+", "", text)

    text = re.sub(r"(?<!\w)\*+|\*+(?!\w)", "", text)
    text = mark_quotes(text)

    return detokenizer.detokenize(text.split())


def length_match(df, tolerance=0.2):
    # HC3: pair each ChatGPT answer with the closest-length human answer to the same question,
    # within 20% of the human answer's length. Anything left unpaired is dropped.
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
    # Used on both corpora: normalise, drop duplicates (normalising can create new ones),
    # then drop anything too short to judge. The class balance is left alone, since the
    # metric looks at human and machine scores separately.
    df = df.assign(text=df["text"].apply(normalise_text))
    df = df.drop_duplicates(subset="text")
    n_tokens = df["text"].str.split().str.len()

    return df[n_tokens >= min_tokens].reset_index(drop=True)


def train_val_split(df, val_fraction=0.15, seed=0):
    # HC3: hold back 15% for setting the threshold only, with the same human/ChatGPT mix in both parts
    train_df, val_df = train_test_split(df, test_size=val_fraction, stratify=df["label"], random_state=seed)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def closest_human(humans_by_length, length, tolerance):
    # Look at the same length first, then one shorter, one longer, two shorter, and so on,
    # and take the first unused human text that's within tolerance
    for gap in range(math.ceil(length * tolerance / (1 - tolerance)) + 1):
        for human_len in (length - gap, length + gap):
            if abs(length - human_len) <= tolerance * human_len and humans_by_length.get(human_len):
                return humans_by_length[human_len].pop()

    return None


def length_match_by_domain(df, tolerance=0.2, seed=0):
    # MAGE: pair each machine text with an unused human text from the same domain, within 20%
    # of the human text's length. Machine texts go in a random order so no family gets first pick.
    n_words = df["text"].str.split().str.len().to_dict()
    rng = np.random.default_rng(seed)
    keep = []

    for _, domain_df in df.groupby("domain"):
        # Group this domain's human texts by length, so finding a partner is a quick lookup
        humans_by_length = {}
        for row in domain_df.index[domain_df["label"] == 0]:
            humans_by_length.setdefault(n_words[row], []).append(row)

        for row in rng.permutation(domain_df.index[domain_df["label"] == 1].to_numpy()):
            partner = closest_human(humans_by_length, n_words[row], tolerance)
            if partner is not None:
                keep += [partner, row]

    # Rows go human, machine, human, machine..., so each pair is two rows in a row
    matched = df.loc[keep].reset_index(drop=True)
    matched["pair_id"] = matched.index // 2

    return matched
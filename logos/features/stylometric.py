"""
The 14 stylometric features that get added to DistilBERT's classifier head.

Defined in Table D.2. The reasoning behind each choice is in Appendix D's notes
and in sections 3.2 and 3.6; this module only implements them.
"""

import statistics
import nltk
import pandas as pd
import textstat

from nltk import sent_tokenize, word_tokenize
from collections import Counter
from nltk.corpus import stopwords

# NLTK renamed the tokeniser and tagger in version 3.9, so both names are listed.
# cmudict is not needed: textstat counts syllables with pyphen.
NLTK_RESOURCES = [
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("tokenizers/punkt", "punkt"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("corpora/stopwords", "stopwords"),
]

# This order is the order the features reach the classifier, so it must not
# change between training and scoring.
FEATURE_NAMES = [
    "sentence_length_mean",
    "sentence_length_sd",
    "mattr",
    "hapax_rate",
    "stopword_rate",
    "punctuation_rate",
    "noun_rate",
    "verb_rate",
    "past_tense_rate",
    "adverb_rate",
    "pronoun_rate",
    "first_person_rate",
    "contraction_rate",
    "flesch_reading_ease",
]

MATTR_WINDOW = 25

# Listed out rather than taken from string.punctuation, which also holds $ % & *
# and other symbols.
PUNCTUATION = set(".,;:!?-—'\"()[]{}/")

# Penn Treebank tags, from nltk.pos_tag. PAST_TENSE_TAGS is a subset of
# VERB_TAGS, so those two features overlap by construction.
NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
PAST_TENSE_TAGS = {"VBD", "VBN"}
ADVERB_TAGS = {"RB", "RBR", "RBS"}
PRONOUN_TAGS = {"PRP", "PRP$", "WP", "WP$"}

# Penn Treebank has no first-person tag, so this feature needs a word list.
FIRST_PERSON_WORDS = {
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
}
PERSONAL_PRONOUN_TAGS = {"PRP", "PRP$"}

# What the tokeniser leaves behind when it splits a contraction: "don't" becomes
# "do" and "n't". "'s" also catches possessives, which this cannot tell apart.
CONTRACTION_FRAGMENTS = {"n't", "'s", "'re", "'ve", "'ll", "'d", "'m"}


def words_in(tokens):
    """
    Keep the tokens with at least one letter in them.

    Contractions have to survive this, because contraction_rate divides by the
    count it returns.
    """
    return [token for token in tokens if any(n.isalpha() for n in token)]


def sentence_length_stats(sentence_lengths):
    # pstdev, not stdev: a text's sentences are the whole population, so this
    # divides by n rather than n - 1.
    if not sentence_lengths:
        return 0.0, 0.0

    return statistics.fmean(sentence_lengths), statistics.pstdev(sentence_lengths)


def mattr(words, window=MATTR_WINDOW):
    """
    Moving-average type-token ratio [60], computed by the rapid algorithm in
    that paper. Expects words already lowercased. Window choice: Appendix D.
    """
    if not words:
        return 0.0

    # Too short for a window, so it falls back to the plain ratio, which does
    # depend on text length. Appendix D reports how often this happens.
    if len(words) <= window:
        return len(set(words)) / len(words)

    counts = Counter(words[:window])
    total_counts = len(counts)

    for index in range(window, len(words)):
        leaving = words[index - window]
        counts[leaving] -= 1

        # Load-bearing. Counter keeps entries sitting at zero, and len(counts)
        # would then count words that have already left the window.
        if counts[leaving] == 0:
            del counts[leaving]

        counts[words[index]] += 1
        total_counts += len(counts)

    positions = len(words) - window + 1

    return total_counts / positions / window


def hapax_rate(words):
    # Divided by distinct words, not total words. Expects words already
    # lowercased.
    if not words:
        return 0.0

    counts = Counter(words)
    once = sum(1 for count in counts.values() if count == 1)

    return once / len(counts)


def stopword_rate(words, stop_words):
    # The list is passed in because building it once is cheap and building it
    # per text is not. Expects words already lowercased.
    if not words:
        return 0.0

    return sum(1 for word in words if word in stop_words) / len(words)


def punctuation_rate(text):
    # Takes the raw text, not the word list, and divides by characters rather
    # than words. The only feature of the fourteen that does.
    if not text:
        return 0.0

    return sum(1 for char in text if char in PUNCTUATION) / len(text)


def tag_rate(tagged, wanted):
    if not tagged:
        return 0.0

    return sum(1 for _, tag in tagged if tag in wanted) / len(tagged)


def first_person_rate(tagged):
    # Needs both the word list and the pronoun tag. Without lower() every
    # capitalised match is missed, and "I" is always capitalised.
    if not tagged:
        return 0.0

    return sum(
        1
        for word, tag in tagged
        if word.lower() in FIRST_PERSON_WORDS and tag in PERSONAL_PRONOUN_TAGS
    ) / len(tagged)


def contraction_rate(words):
    # Expects words already lowercased.
    if not words:
        return 0.0

    return sum(1 for word in words if word in CONTRACTION_FRAGMENTS) / len(words)


def flesch_reading_ease(words, sentence_count):
    """
    Flesch reading ease in the form given in [63]. Not a proportion: it runs
    from about -100 to +120 and is deliberately not clamped. Appendix D records
    how this implementation compares with Flesch's own published scores.
    """
    if not words or sentence_count == 0:
        return 0.0

    syllables = sum(textstat.syllable_count(word) for word in words)
    word_length = syllables / len(words) * 100
    syllable_length = len(words) / sentence_count

    return 206.835 - 0.846 * word_length - 1.015 * syllable_length


def stylometric(text, stop_words=None):
    """All 14 features for one text, keyed by FEATURE_NAMES."""
    if stop_words is None:
        stop_words = set(stopwords.words("english"))

    # Sentences first, then words within each sentence, so the sentence lengths
    # and the flat word list come from one pass and cannot disagree.
    sentence_tokens = [word_tokenize(sentence) for sentence in sent_tokenize(text)]
    sentence_words = [words_in(tokens) for tokens in sentence_tokens]
    sentence_lengths = [len(sentence) for sentence in sentence_words if sentence]
    words = [word for sentence in sentence_words for word in sentence]
    lowercase_words = [word.lower() for word in words]

    # Tag with the punctuation still in place, then drop it. The tagger uses
    # nearby tokens as context, so stripping full stops first makes it mistake
    # where sentences start.
    tagged = [
        (word, tag)
        for tokens in sentence_tokens
        for word, tag in nltk.pos_tag(tokens)
        if any(char.isalpha() for char in word)
    ]

    # All fourteen zero is the signal that a text had no usable words. Without
    # this, punctuation_rate would still return a value because it reads the raw
    # text, and run_stylometric_features.py counts these rows.
    if not words:
        return dict.fromkeys(FEATURE_NAMES, 0.0)

    length_mean, length_sd = sentence_length_stats(sentence_lengths)

    return {
        "sentence_length_mean": length_mean,
        "sentence_length_sd": length_sd,
        "mattr": mattr(lowercase_words),
        "hapax_rate": hapax_rate(lowercase_words),
        "stopword_rate": stopword_rate(lowercase_words, stop_words),
        "punctuation_rate": punctuation_rate(text),
        "noun_rate": tag_rate(tagged, NOUN_TAGS),
        "verb_rate": tag_rate(tagged, VERB_TAGS),
        "past_tense_rate": tag_rate(tagged, PAST_TENSE_TAGS),
        "adverb_rate": tag_rate(tagged, ADVERB_TAGS),
        "pronoun_rate": tag_rate(tagged, PRONOUN_TAGS),
        "first_person_rate": first_person_rate(tagged),
        "contraction_rate": contraction_rate(lowercase_words),
        "flesch_reading_ease": flesch_reading_ease(lowercase_words, len(sentence_lengths)),
    }


def features_frame(texts, show_progress=False):
    """All 14 features for a list of texts, one row each."""
    stop_words = set(stopwords.words("english"))
    rows = []

    for index, text in enumerate(texts):
        if show_progress and index and index % 1000 == 0:
            print(f"  {index:,} of {len(texts):,}", flush=True)

        rows.append(stylometric(text, stop_words))

    # Columns are picked by name, so a name missing from FEATURE_NAMES is
    # silently dropped.
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def ensure_nltk_data():
    """
    Download whatever NLTK data is missing. Safe to call repeatedly, and worth
    calling on a login node, because compute nodes may have no network access.
    """
    missing = []

    for path, name in NLTK_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            missing.append(name)

    for name in missing:
        nltk.download(name, quiet=True)

"""
The 14 stylometric features that get added to DistilBERT's classifier head.

Each feature is one number per text. None of them measures how long the text is,
because the training pairs are length-matched and length is not a signal this
detector is allowed to use.

Thirteen of them are proportions between 0 and 1. Flesch reading ease is not, so
the features have to be standardised before they are joined onto DistilBERT's
output, with the standardiser fitted on training data only.
"""

import statistics
import nltk
import pandas as pd
import textstat

from nltk import sent_tokenize, word_tokenize
from collections import Counter
from nltk.corpus import stopwords

# NLTK keeps its tokeniser, tagger and word lists outside the package, and
# renamed two of them in version 3.9, so both the old and new names are listed.
# textstat counts syllables with pyphen, so cmudict is not needed.
NLTK_RESOURCES = [
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("tokenizers/punkt", "punkt"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("corpora/stopwords", "stopwords"),
]

# The order here is the order the features are fed to the classifier, so it must
# stay the same between training and scoring.
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

# 25 is small, but the corpus keeps texts down to ten words, and a bigger window
# would send too many of them down the fallback branch in mattr().
MATTR_WINDOW = 25

# Listed out rather than taken from string.punctuation, which also holds $ % & *
# and friends. Those are symbols, so a price or an email would read as heavy
# punctuation.
PUNCTUATION = set(".,;:!?-—'\"()[]{}/")

# Penn Treebank tags, which is what nltk.pos_tag gives back. Note that
# past_tense_tags is a subset of verb_tags, so those two features overlap on
# purpose. Modals (can, will, should) are tagged MD and so do not count as verbs.
NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
PAST_TENSE_TAGS = {"VBD", "VBN"}
ADVERB_TAGS = {"RB", "RBR", "RBS"}
PRONOUN_TAGS = {"PRP", "PRP$", "WP", "WP$"}

# Penn Treebank has no first-person tag: PRP covers I, you, he, she, it, we and
# they together. So this one needs a word list. The tag is checked as well, which
# keeps "the gold mine" from counting and keeps this feature a subset of
# pronoun_rate.
FIRST_PERSON_WORDS = {
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
}
PERSONAL_PRONOUN_TAGS = {"PRP", "PRP$"}

# The tokeniser splits contractions, so "don't" arrives as "do" and "n't". These
# are the pieces it leaves behind. Note that "'s" also catches possessives, so
# "the cat's bowl" counts as a contraction.
CONTRACTION_FRAGMENTS = {"n't", "'s", "'re", "'ve", "'ll", "'d", "'m"}


def words_in(tokens):
    """
    Keep the tokens that count as words.

    A token needs at least one letter, so full stops and bare numbers are
    dropped while words like "don't" are kept. Keeping contractions matters,
    because contraction_rate divides by this count.
    """
    return [token for token in tokens if any(n.isalpha() for n in token)]


def sentence_length_stats(sentence_lengths):
    # pstdev, not stdev. A text's sentences are the whole thing being measured,
    # not a sample of something bigger, so this divides by n rather than n - 1.
    if not sentence_lengths:
        return 0.0, 0.0

    return statistics.fmean(sentence_lengths), statistics.pstdev(sentence_lengths)


def mattr(words, window=MATTR_WINDOW):
    """
    Moving-average type-token ratio (Covington and McFall, 2010).

    The plain type-token ratio drops as a text gets longer, because the further
    you read the likelier the next word is one already used. So it partly
    measures length. This takes the ratio inside a window of fixed size, slides
    the window along one word at a time, and averages. Every window is the same
    size, so the answer does not depend on the length of the text.
    """
    if not words:
        return 0.0

    # Too short for a window, so fall back to the plain ratio. That version does
    # depend on length, which is worth remembering when reading the results.
    if len(words) <= window:
        return len(set(words)) / len(words)

    counts = Counter(words[:window])
    total_counts = len(counts)

    for index in range(window, len(words)):
        leaving = words[index - window]
        counts[leaving] -= 1

        # This delete is doing real work. Counter keeps an entry sitting at zero,
        # and len(counts) would then count words that have already left.
        if counts[leaving] == 0:
            del counts[leaving]

        counts[words[index]] += 1
        total_counts += len(counts)

    positions = len(words) - window + 1

    return total_counts / positions / window


def hapax_rate(words):
    """
    The share of a text's distinct words that it uses exactly once.

    Divided by distinct words, not total words. The per-word version falls
    steeply as texts get longer, so it ends up measuring length.
    """
    if not words:
        return 0.0

    counts = Counter(words)
    once = sum(1 for count in counts.values() if count == 1)

    return once / len(counts)


def stopword_rate(words, stop_words):
    # The list is passed in rather than built here, because building it once is
    # cheap and building it 392,385 times is not.
    if not words:
        return 0.0

    return sum(1 for word in words if word in stop_words) / len(words)


def punctuation_rate(text):
    # Counted over characters, including spaces, rather than over words. Done on
    # the raw text so the answer does not depend on how the tokeniser split
    # things up. This is the only feature that works this way.
    if not text:
        return 0.0

    return sum(1 for char in text if char in PUNCTUATION) / len(text)


def tag_rate(tagged, wanted):
    """What share of the tagged words carry one of the tags being looked for."""
    if not tagged:
        return 0.0

    return sum(1 for _, tag in tagged if tag in wanted) / len(tagged)


def first_person_rate(tagged):
    # The word has to be in the list AND be tagged as a pronoun. Without the
    # lower() every capitalised one is missed, and "I" is always capitalised.
    if not tagged:
        return 0.0

    return sum(
        1
        for word, tag in tagged
        if word.lower() in FIRST_PERSON_WORDS and tag in PERSONAL_PRONOUN_TAGS
    ) / len(tagged)


def contraction_rate(words):
    if not words:
        return 0.0

    return sum(1 for word in words if word in CONTRACTION_FRAGMENTS) / len(words)


def flesch_reading_ease(words, sentence_count):
    """
    Flesch reading ease, in the form given in Flesch (1948).

    Higher means easier to read. The score is not a proportion: it runs from
    about -100 to +120, and is deliberately not clamped, because short simple
    sentences can legitimately score above 100.

    Syllables come from textstat, and words and sentences are counted by this
    study's own rules rather than Flesch's manual ones, which treated figures as
    words. On the two sample passages in the paper this scores 48.2 and 68.4
    against his published 46 and 66.
    """
    if not words or sentence_count == 0:
        return 0.0

    syllables = sum(textstat.syllable_count(word) for word in words)
    word_length = syllables / len(words) * 100
    syllable_length = len(words) / sentence_count

    return 206.835 - 0.846 * word_length - 1.015 * syllable_length


def stylometric(text, stop_words=None):
    """
    All 14 features for one text, as a dictionary.

    A text with no words in it comes back as all zeros rather than raising, so
    one bad row cannot stop a corpus-wide run.
    """
    if stop_words is None:
        stop_words = set(stopwords.words("english"))

    # Split into sentences first, then words within each sentence, so the
    # sentence lengths and the flat word list come from one pass and cannot
    # disagree with each other.
    sentence_tokens = [word_tokenize(sentence) for sentence in sent_tokenize(text)]
    sentence_words = [words_in(tokens) for tokens in sentence_tokens]
    sentence_lengths = [len(sentence) for sentence in sentence_words if sentence]
    words = [word for sentence in sentence_words for word in sentence]
    lowercase_words = [word.lower() for word in words]

    # Tag each sentence with its punctuation still in place, then drop the
    # punctuation from the result. The tagger uses nearby tokens as context, so
    # removing full stops first makes it guess sentence starts wrongly.
    tagged = [
        (word, tag)
        for tokens in sentence_tokens
        for word, tag in nltk.pos_tag(tokens)
        if any(char.isalpha() for char in word)
    ]

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
    """All 14 features for a list of texts, one row each.

    The stop-word list is built once here rather than once per text, which is
    the reason stylometric() takes it as an argument.
    """
    stop_words = set(stopwords.words("english"))
    rows = []

    for index, text in enumerate(texts):
        if show_progress and index and index % 1000 == 0:
            print(f"  {index:,} of {len(texts):,}", flush=True)

        rows.append(stylometric(text, stop_words))

    # Columns are selected by name, so FEATURE_NAMES fixes the order the
    # classifier sees. Any name missing from that list is silently dropped.
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def ensure_nltk_data():
    """Download whatever NLTK data is missing.

    Safe to call more than once. Worth calling on a login node before
    submitting a job, because compute nodes may have no network access.
    """
    missing = []

    for path, name in NLTK_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            missing.append(name)

    for name in missing:
        nltk.download(name, quiet=True)

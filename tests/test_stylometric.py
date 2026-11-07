"""Tests for the 14 stylometric features.

Every expected value is either worked out by hand from a short text, or taken
from the paper the measure comes from. Two are checks against published
figures rather than against my own arithmetic:

  - MATTR against the worked example on page 97 of Covington and McFall (2010).
  - Flesch reading ease against the two sample passages in Flesch (1948),
    whose published scores are 46 and 66.

Note that mattr, hapax_rate, stopword_rate and contraction_rate expect words
that have already been lowercased, because stylometric() lowercases once and
passes the result to all four. The tests below do the same.
"""

import pytest
from nltk.tokenize import sent_tokenize, word_tokenize

from logos.features.stylometric import (
    FEATURE_NAMES,
    NOUN_TAGS,
    VERB_TAGS,
    contraction_rate,
    ensure_nltk_data,
    features_frame,
    first_person_rate,
    flesch_reading_ease,
    hapax_rate,
    mattr,
    punctuation_rate,
    sentence_length_stats,
    stopword_rate,
    stylometric,
    tag_rate,
    words_in,
)

# Flesch's two sample passages, from pages 231-232, with the punctuation the
# scanned text lost put back. He reports scores of 46 and 66.
FLESCH_LIFE = """Except in the field of surgery, control of pain is still very much in the
primitive stages. Countless thousands of patients suffer the tortures of cancer,
angina pectoris and other distressing diseases while their physicians are helpless
to relieve them. A big step toward help for these sufferers is now being made
with a treatment known as nerve-blocking. This treatment, which consists
of putting a "block" between the source of pain and the brain, is not a new
therapy. But its potentialities are just now being realized. Using better
drugs and a wider knowledge of the mechanics of pain gained during and since
the war, Doctors E. A. Rovenstine and E. M. Papper of the New York University
College of Medicine have been able to help two-thirds of the patients
accepted for treatment in their "pain clinic" at Bellevue Hospital.
The nerve-block treatment is comparatively simple and does not have
serious aftereffects. It merely involves the injection of an anesthetic drug
along the path of the nerve carrying pain impulses from the diseased or injured
tissue to the brain. Although its action is similar to that of spinal anesthesia
used in surgery, nerve block generally lasts much longer and is only occasionally
used for operations. The N. Y. U. doctors have found it effective in a wide
range of diseases, including angina pectoris, sciatica, shingles, neuralgia and
some forms of cancer. Relief is not always permanent, but usually the injection
can be repeated. Some angina pectoris patients have had relief for periods
ranging from six months to two years. While recognizing that nerve block
is no panacea, the doctors feel that results obtained in cases like that of Mike
Ostroich (next page) will mean a much wider application in the near future."""

FLESCH_NEW_YORKER = """Recently, Rovenstine devoted a few minutes to relieving a free
patient in Bellevue of a pain in an arm that had been cut off several years
before. The victim of this phantom pain said that the tendons ached and
that his fingers were clenched so hard he could feel his nails digging into his
palm. Dr. Rovenstine's assistant, Dr. E. M. Papper, reminded Rovenstine
that a hundred and fifty years ago the cure would have been to dig up the
man's arm, if its burial place was known, and straighten out the hand. Rovenstine
smiled. "I tell you," he said. "We'll use a two-percent solution of
procaine, and if it works, in a couple of weeks we'll go on with an alcohol
solution. Procaine, you know, lasts a couple of weeks, alcohol six months or
longer. In most cases of this sort, I use the nerve block originated by Labat
around 1910 and improved on in New Orleans about ten years back, plus one
or two improvisations of my own." (Nerve blocking is a method of anesthetizing
a nerve that is transmitting pain.)
The man with the pain in the nonexistent hand was an indigent, and
Rovenstine was working before a large gallery of student anesthetists and
visitors when he exorcised the ghosts that were paining him. Some of the
spectators, though they felt awed, also felt inclined to giggle. Even trained
anesthetists sometimes get into this state during nerve-block demonstrations
because of the tenseness such feats of magic induce in them. The patient,
thin, stark-naked, and an obvious product of poverty and cheap gin mills,
was nervous and rather apologetic when he was brought into the operating
theatre. He lay face down on the operating table. Rovenstine has an easy
manner with patients, and as his thick, stubby hands roamed over the man's
back, he gently asked, "How you doing?" "My hand, it is all closed together,
Doc," the man answered, startled and evidently a little proud of the attention
he was getting. "You'll be O.K. soon," Rovenstine said, and turned to the
audience. "One of my greatest contributions to medical science has been the
use of the eyebrow pencil," he said. He took one from the pocket of his white
smock and made a series of marks on the patient's back, near the shoulder of
the amputated arm, so that the spectators could see exactly where he was
going to work. With a syringe and needle, he raised four small weals on the
man's back and then shoved long needles into the weals. The man shuddered
but said he felt no pain. Rovenstine then attached a syringe to the first
needle, injected the procaine solution, unfastened the syringe, attached it to
the next needle, injected more of the solution, and so on. The patient's face
began to relax a little. "Lord, Doc," he said. "My hand is loosening up a
bit already." "You'll be all right by tonight, I think," Rovenstine said.
He was."""


@pytest.fixture(scope="session", autouse=True)
def nltk_data():
    """The tokeniser, tagger and word lists live outside the package."""
    ensure_nltk_data()


def test_words_in_keeps_letters_only():
    assert words_in(word_tokenize("The cat sat on the mat.")) == [
        "The", "cat", "sat", "on", "the", "mat"]

    # A contraction survives in pieces, and both pieces have letters, so both
    # count as words. contraction_rate depends on that.
    assert words_in(word_tokenize("I don't.")) == ["I", "do", "n't"]

    # Bare numbers go, alphanumerics stay
    assert words_in(word_tokenize("COVID-19 affected 2,500 people.")) == [
        "COVID-19", "affected", "people"]

    # The maths expression found in HC3. No letters, so no words.
    assert words_in(word_tokenize("2 ^ 2 / 2 ^ 2 = 1")) == []


def test_sentence_length_uses_the_population_deviation():
    """Sentences of three and five words. The mean is 4. The population
    deviation divides by n and gives 1.0; the sample version would give
    1.414, so this pins down which one is used."""
    assert sentence_length_stats([3, 5]) == pytest.approx((4.0, 1.0))
    assert sentence_length_stats([6]) == pytest.approx((6.0, 0.0))
    assert sentence_length_stats([]) == (0.0, 0.0)


def test_mattr_reproduces_the_worked_example_in_the_paper():
    """Covington and McFall, page 97. Both sequences have the same plain
    type-token ratio, but with a small window the first rises to 1 and the
    second does not, and with a large enough window they converge."""
    first = "a b c d e f g a b c d e f g".split()
    second = "a a b b c c d d e e f f g g".split()

    assert len(set(first)) / len(first) == pytest.approx(0.5)
    assert len(set(second)) / len(second) == pytest.approx(0.5)

    assert mattr(first, window=7) == pytest.approx(1.0)
    assert mattr(second, window=7) == pytest.approx(4 / 7)
    assert mattr(first, window=13) == pytest.approx(mattr(second, window=13))


def test_mattr_bounds_and_short_text_fallback():
    assert mattr([f"w{i}" for i in range(120)]) == pytest.approx(1.0)
    assert mattr(["a"] * 120) == pytest.approx(1 / 25)
    # Below the window, so the plain ratio over the whole text
    assert mattr([f"w{i}" for i in range(10)]) == pytest.approx(1.0)
    assert mattr([]) == 0.0


def test_hapax_counts_types_not_tokens():
    # Four types, three of them used once
    assert hapax_rate(["a", "b", "c", "d", "d"]) == pytest.approx(0.75)
    assert hapax_rate(["a", "a"]) == pytest.approx(0.0)
    assert hapax_rate([]) == 0.0


def test_stopword_rate_over_words():
    stop_words = {"the", "on"}
    words = "the cat sat on the mat".split()
    assert stopword_rate(words, stop_words) == pytest.approx(3 / 6)
    assert stopword_rate([], stop_words) == 0.0


def test_punctuation_rate_is_over_characters():
    # 23 characters including the spaces, one full stop
    assert punctuation_rate("The cat sat on the mat.") == pytest.approx(1 / 23)
    assert punctuation_rate("No punctuation here") == 0.0
    assert punctuation_rate("") == 0.0


def test_every_feature_of_one_short_text_by_hand():
    """"The cat sat on the mat." worked out by hand.

    Six words, one sentence, 23 characters. Lowercased types are the, cat,
    sat, on, mat, so five types with four used once. Stop words are the, on,
    the. Tagging gives cat and mat as nouns and sat as a past-tense verb.
    Every word is one syllable, so wl is 100 and sl is 6.
    """
    features = stylometric("The cat sat on the mat.")

    assert list(features) == FEATURE_NAMES
    assert features["sentence_length_mean"] == pytest.approx(6.0)
    assert features["sentence_length_sd"] == pytest.approx(0.0)
    assert features["mattr"] == pytest.approx(5 / 6)
    assert features["hapax_rate"] == pytest.approx(4 / 5)
    assert features["stopword_rate"] == pytest.approx(3 / 6)
    assert features["punctuation_rate"] == pytest.approx(1 / 23)
    assert features["noun_rate"] == pytest.approx(2 / 6)
    assert features["verb_rate"] == pytest.approx(1 / 6)
    assert features["past_tense_rate"] == pytest.approx(1 / 6)
    assert features["adverb_rate"] == pytest.approx(0.0)
    assert features["pronoun_rate"] == pytest.approx(0.0)
    assert features["first_person_rate"] == pytest.approx(0.0)
    assert features["contraction_rate"] == pytest.approx(0.0)
    # 206.835 - 0.846 * 100 - 1.015 * 6
    assert features["flesch_reading_ease"] == pytest.approx(116.145, abs=1e-3)


def test_first_person_needs_the_lowercase_check():
    """I is always capitalised in English, so without lowercasing before the
    word-list check this feature misses the commonest first-person pronoun
    there is. That bug was live until it was caught on the Declaration of
    Independence, where the only first-person word is a capitalised "We"."""
    features = stylometric("I think we should go, it was our idea.")
    # I, we, our among nine words
    assert features["first_person_rate"] == pytest.approx(3 / 9)


def test_first_person_requires_a_pronoun_tag():
    """The word list alone would count "mine" in "the gold mine". The tag check
    is what stops it."""
    assert stylometric("The gold mine was deep.")["first_person_rate"] == 0.0


def test_the_overlapping_features_are_subsets():
    """Two overlaps hold by construction, and section 3.5 reports them."""
    features = stylometric(
        "I had walked there before. We have seen it ourselves, and I think "
        "they had already gone when she arrived."
    )
    assert features["past_tense_rate"] <= features["verb_rate"]
    assert features["first_person_rate"] <= features["pronoun_rate"]


def test_modals_do_not_count_as_verbs():
    """Penn Treebank tags can and will as MD, which is not in VERB_TAGS. Six
    words, two of them tagged as verbs, so this records the decision."""
    assert stylometric("She can run and will win.")["verb_rate"] == pytest.approx(2 / 6)


def test_contraction_rate_counts_tokeniser_fragments():
    """Fifteen words, and the tokeniser leaves n't, 's, 'd and n't behind. Note
    that the 's here is a contraction, but the same fragment is left by a
    possessive, which this feature cannot tell apart."""
    text = "I can't say it's ours; they'd have told us, wouldn't they?"
    assert stylometric(text)["contraction_rate"] == pytest.approx(4 / 15)

    # A possessive counts too, which is the known limitation
    assert stylometric("The cat's bowl is empty.")["contraction_rate"] > 0


@pytest.mark.parametrize(
    "text, published, expected", [(FLESCH_LIFE, 46, 48.2), (FLESCH_NEW_YORKER, 66, 68.4)])
def test_flesch_matches_the_published_sample_scores(text, published, expected):
    """Flesch's two sample passages, for which he publishes 46 and 66.

    This counts words and sentences by this study's rules rather than Flesch's
    manual ones, and takes syllables from textstat, so the agreement is close
    rather than exact. The error runs the same way both times, which is the
    point: a steady offset rather than anything erratic. Appendix D quotes the
    expected figures, so a change here means Appendix D needs updating.
    """
    words = [word.lower() for word in words_in(word_tokenize(text))]
    score = flesch_reading_ease(words, len(sent_tokenize(text)))

    assert abs(score - published) < 3.0
    assert score == pytest.approx(expected, abs=0.5)


@pytest.mark.parametrize("text", ["", "   ", "...", "2 ^ 2 = 4"])
def test_a_text_with_no_words_gives_zeros(text):
    assert set(stylometric(text).values()) == {0.0}


@pytest.mark.parametrize("text", ["The cat sat on the mat.", "Hello.", FLESCH_LIFE])
def test_no_feature_is_nan_or_infinite(text):
    """A nan would poison the standardiser and then all 782 inputs to the
    classifier head."""
    for name, value in stylometric(text).items():
        assert value == value, f"{name} is nan"
        assert abs(value) != float("inf"), f"{name} is infinite"


def test_tag_rate_handles_an_empty_list():
    assert tag_rate([], NOUN_TAGS) == 0.0
    assert tag_rate([("cat", "NN"), ("ran", "VBD")], VERB_TAGS) == pytest.approx(0.5)


def test_features_frame_columns_and_rows():
    frame = features_frame(["The cat sat on the mat.", "Hello there."])
    assert list(frame.columns) == FEATURE_NAMES
    assert len(frame) == 2
    assert frame["sentence_length_mean"].iloc[0] == pytest.approx(6.0)

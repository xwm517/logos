from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def lexical_baseline(C=1.0, seed=0):
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=2**21),
        LogisticRegression(C=C, solver="liblinear", class_weight="balanced", random_state=seed),
    )
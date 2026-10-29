from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# The C values searched by OpenAI's GPT-2 output baseline [52], [53]
c_grid = [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64]


def lexical_baseline(C=1.0, seed=0):
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=2**21),
        LogisticRegression(C=C, solver="liblinear", class_weight="balanced", random_state=seed),
    )


def fit_lexical_baseline(train_texts, train_labels, tune_texts, tune_labels, seed=0, c_grid=c_grid):
    # Try every C and keep the model that scores best on the tuning set
    best_model, best_c, best_accuracy = None, None, -1
    accuracies = {}

    for c in c_grid:
        model = lexical_baseline(C=c, seed=seed).fit(train_texts, train_labels)
        accuracies[c] = model.score(tune_texts, tune_labels)

        if accuracies[c] > best_accuracy:
            best_model, best_c, best_accuracy = model, c, accuracies[c]

    return best_model, best_c, accuracies
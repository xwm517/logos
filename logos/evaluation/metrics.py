import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_ind


def calibrate_threshold(scores, labels, target_fpr=0.01):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    human = np.sort(scores[labels == 0])
    k = int(np.floor(target_fpr * len(human)))

    return float(np.nextafter(human[len(human) - k - 1], np.inf))


def tpr_at_threshold(scores, labels, tau):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    machine = scores[labels == 1]

    return float(np.mean(machine >= tau))


def fpr_at_threshold(scores, labels, tau):
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
    human = scores[labels == 0]

    return float(np.mean(human >= tau))


def auroc(scores, labels):
    return float(roc_auc_score(labels, scores))


def cohens_d(values, labels):
    human = values[labels == 0]
    machine = values[labels == 1]

    n_human, n_machine = len(human), len(machine)
    pooled = np.sqrt(
        ((n_human - 1) * human.var(ddof=1) + (n_machine - 1) * machine.var(ddof=1))
        / (n_human + n_machine - 2)
    )

    if pooled == 0:
        return 0.0

    return (machine.mean() - human.mean()) / pooled


def welch_p(a, b):
    return ttest_ind(a, b, equal_var=False).pvalue
import numpy as np


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
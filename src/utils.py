import math

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def get_pair_accuracy(similarities, threshold, labels):
    """

    :param similarities: list of similarity scores
    :param threshold: threshold to be used
    :param labels: same/different 0/1 labels
    :return:
    """
    s = np.array(similarities, dtype=np.float32)
    l = np.array(labels, dtype=np.float32)
    t = threshold
    acc = float(len(np.where(s[np.where(l == 1)[0]] > t)[0]) + len(np.where(s[np.where(l != 1)[0]] < t)[0])) / len(s)

    # number of ocasions where the pairs are different and predicted to be the same
    FP = float(len(np.where(s[np.where(l != 1)[0]] > t)[0]))
    # number of ocasions where the pairs are the same and predicted to be different
    FN = float(len(np.where(s[np.where(l == 1)[0]] < t)[0]))
    # number of ocasions where the pairs are the same and predicted to be the same
    TP = float(len(np.where(s[np.where(l == 1)[0]] > t)[0]))
    # number of ocasions where the pairs are different and predicted to be different
    TN = float(len(np.where(s[np.where(l != 1)[0]] < t)[0]))

    accuracy = float((TP + TN) / len(s))
    TPR = TP / (TP + FN) if (TP + FN) != 0 else float("inf")
    TNR = TN / (TN + FP) if (TN + FP) != 0 else float("inf")
    FPR = FP / (FP + TN) if (FP + TN) != 0 else float("inf")
    FNR = FN / (TP + FN) if (TP + FN) != 0 else float("inf")
    return acc, TPR, TNR, FPR, FNR


def run_10_fold_test(featuresL, featuresR, labels):
    """
    Calculates different measurements on LFW benchmark using the default 10 fold cross validation

    :param featuresL: Left features
    :param featuresR: Right features
    :param labels: (int) list of labels (same=1/different=0)
    :return: accuracy, TPR, TNR, FPR, FNR
    """

    thresholds = np.linspace(0, math.pi, 50)
    labels = np.array(labels)
    kf = KFold(n_splits=10, shuffle=False)
    performance = {"accuracy": [],
                   "TPR": [],
                   "TNR": [],
                   "FPR": [],
                   "FNR": [],
                   "threshold": []}

    for train_index, test_index in tqdm(kf.split(labels)):
        # Calculate cosine similarity and take the diagonal. Note: very inefficient.
        similarity = np.diag(cosine_similarity(featuresL, featuresR))
        # calculate accuracy for each threshold.
        acc = [get_pair_accuracy(similarity[train_index], t, labels[train_index])[0] for t in thresholds]
        # select the best threshold based on training set.
        threshold = np.mean(thresholds[np.where(acc == np.max(acc))[0]])
        # measure performance on the test set.
        acc, TPR, TNR, FPR, FNR = get_pair_accuracy(similarity[test_index], threshold, labels[test_index])
        performance["accuracy"].append(acc)
        performance["TPR"].append(TPR)
        performance["TNR"].append(TNR)
        performance["FPR"].append(FPR)
        performance["FNR"].append(FNR)
        performance["threshold"].append(threshold)

    return performance["accuracy"], performance["TPR"], performance["TNR"], performance["FPR"], performance["FNR"]


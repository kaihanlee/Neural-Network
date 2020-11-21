import numpy as np


def compute_average_accuracy(predictions, targets):
    return np.mean(predictions == targets)

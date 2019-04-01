import pandas as pd
import numpy as np


def get_accuracy(y_pred, y_test, percentile=1.0):
    """
        y_pred: np.array, first column is probability for negative classification
                          second column is probability for negative classification
        y_test: np.array, an array of 0 and 1
        percentile: float, 0 <= percentile <= 1
    """
    assert 0. <= percentile <= 1., "percentile should be between 0 and 1"
    df = pd.DataFrame()
    df['pred'] = (y_pred[:, 1] - 0.5) * 2
    df['test'] = (y_test - 0.5) * 2
    upper = df['pred'].quantile(1 - percentile / 2.)
    lower = df['pred'].quantile(percentile / 2.)
    df = df[(df['pred'] >= upper) | (df['pred'] <= lower)]
    correct = df[np.sign(df['pred']) == np.sign(df['test'])]
    return correct.shape[0] / float(df.shape[0])

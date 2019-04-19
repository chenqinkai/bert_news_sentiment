import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
import math


def mcc(tp, tn, fp, fn):
    up = tp * tn - fp * fn
    t1 = (tp + fp) if (tp + fp) else 1
    t2 = (tp + fn) if (tp + fn) else 1
    t3 = (tn + fp) if (tn + fp) else 1
    t4 = (tn + fn) if (tn + fn) else 1
    down = math.sqrt(t1 * t2 * t3 * t4)
    return float(up) / down


def get_accuracy(y_pred, y_test, percentile=1.0, transform=True, calculate_mcc=False):
    """
        y_pred: np.array, first column is probability for negative classification
                          second column is probability for negative classification
        y_test: np.array, an array of 0 and 1
        percentile: float, 0 <= percentile <= 1
        transform: bool, if we should transform data format passed from y_pred and y_test
    """
    assert 0. <= percentile <= 1., "percentile should be between 0 and 1"
    df = pd.DataFrame()
    df['pred'] = (y_pred[:, 1] - 0.5) * 2 if transform else y_pred
    df['test'] = (y_test - 0.5) * 2 if transform else y_test
    upper = df['pred'].quantile(1 - percentile / 2.)
    lower = df['pred'].quantile(percentile / 2.)
    df = df[(df['pred'] >= upper) | (df['pred'] <= lower)]
    correct = df[np.sign(df['pred']) == np.sign(df['test'])]
    accuracy = correct.shape[0] / float(df.shape[0])
    if calculate_mcc:
        tp = df[(np.sign(df['pred']) >= 0) & (
            np.sign(df['test']) >= 0)].shape[0]
        tn = df[(np.sign(df['pred']) < 0) & (
            np.sign(df['test']) <= 0)].shape[0]
        fp = df[(np.sign(df['pred']) >= 0) & (
            np.sign(df['test']) < 0)].shape[0]
        fn = df[(np.sign(df['pred']) < 0) & (
            np.sign(df['test']) >= 0)].shape[0]
        mcc_value = mcc(tp, tn, fp, fn)
        return accuracy, mcc_value
    return accuracy


def load_prediction(prediction_file_path):
    """
        load prediction into a list from tsv
    """
    if prediction_file_path.endswith(".tsv"):
        with open(prediction_file_path, 'r') as f:
            df_pred = pd.read_csv(f, sep='\t', header=None)
        return ((df_pred[1] - 0.5) * 2).tolist()
    elif prediction_file_path.endswith(".csv"):
        with open(prediction_file_path, 'r') as f:
            df_pred = pd.read_csv(f, header=None)
        return ((df_pred[1] - 0.5) * 2).tolist()
    elif prediction_file_path.endswith(".npy"):
        df_pred = np.load(prediction_file_path)
        return (df_pred[:, 1] - 0.5) * 2
    else:
        raise NotImplementedError("only .tsv, .csv and .npy are accepted")


def load_test_tsv(path):
    """
        load test tsv, then move news in weekend to its next business day
    """
    # path = r"D:\data\reuters_headlines_by_ticker\horizon_3\test_horizon_3.tsv"
    df = pd.read_csv(path, sep='\t')
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df["Date"] = df["Date"].apply(
        lambda x: x + BDay(1) if x.weekday() >= 5 else x)
    # df = df.apply(move_weekend)
    return df

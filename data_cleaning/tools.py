import os

import numpy as np
import pandas as pd

TWEET_WITH_RETURN_DIR = "D:/data/bert_news_sentiment/tweet_with_return"


def cat_all_csv(file_dir, save_path):
    df_all = []
    for f in os.listdir(file_dir):
        print(f)
        df = pd.read_csv(os.path.join(file_dir, f), index_col=0)
        df_all += [df]
    df_all = pd.concat(df_all)
    df_all.index = range(df_all.shape[0])
    df_all.to_csv(save_path)
    return df_all


def cat_all_csv_rt(file_dir, save_path):
    df_all = []
    for year in os.listdir(file_dir):
        # debug
        # if year != "2018":
        #     continue
        for f in os.listdir(os.path.join(file_dir, year)):
            print(f)
            path = os.path.join(file_dir, year, f)
            df = pd.read_csv(path, header=None)
            df_all += [df]
    df_all = pd.concat(df_all)
    df_all.to_csv(save_path, index=False)
    return df_all


def label_training_data(df_news, col_name, percentile):
    # deal with cases where \t is part of headline
    # in this case, it will sabotage the tsv format
    df_news["Headline"] = df_news["Headline"].str.replace("\t", " ")
    df_news["CleanHeadline"] = df_news["CleanHeadline"].str.replace("\t", " ")
    upper_threshold = df_news[col_name].quantile(1 - percentile)
    lower_threshold = df_news[col_name].quantile(percentile)
    df_positive = df_news[df_news[col_name] > upper_threshold]
    df_positive["Label"] = 1
    df_negative = df_news[df_news[col_name] < lower_threshold]
    df_negative["Label"] = 0
    return pd.concat([df_positive, df_negative]).sort_values("Date")


def label_test_data(df_news, col_name):
    df_news["Headline"] = df_news["Headline"].str.replace("\t", " ")
    df_news["CleanHeadline"] = df_news["CleanHeadline"].str.replace("\t", " ")
    df_positive = df_news[df_news[col_name] > 0]
    df_positive["Label"] = 1
    df_negative = df_news[df_news[col_name] < 0]
    df_negative["Label"] = 0
    return pd.concat([df_negative, df_positive]).sort_values("Date")


def get_accuracy(df_news, predict_col_name, prediction, percentile):
    """
        type(prediction): list, which contains the score of the news
    """
    assert df_news.shape[0] == len(prediction)
    df_news["prediction"] = prediction
    th_negative = np.percentile(prediction, percentile / 2)
    th_positive = np.percentile(prediction, 100 - percentile / 2)
    df_partial = df_news[(df_news["prediction"] < th_negative) | (
        df_news["prediction"] > th_positive)]
    total = df_partial.shape[0]
    correct = df_partial[np.sign(df_partial[predict_col_name]) == np.sign(
        df_partial["prediction"])].shape[0]
    return correct / float(total)


from matplotlib import pyplot as plt


def plot_accuracy(df_news, predict_col_name, prediction):
    s = pd.Series()
    for p in range(1, 99):
        s.set_value(p, get_accuracy(df_news, predict_col_name, prediction, p))
    plt.plot(s)

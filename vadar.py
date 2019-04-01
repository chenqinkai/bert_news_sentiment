from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
from utils import get_accuracy
from nltk.corpus import stopwords
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd


if os.name == "nt":
    DATA_DIR = r"D:\data\reuters_headlines_by_ticker\horizon_3"
elif os.name == "posix":
    DATA_DIR = "/home/chenqinkai/data/bert-news-sentiment/news/reuters/horizon_3"
TRAIN_TSV_NAME = "training_horizon_3_percentile_10.tsv"
TEST_TSV_NAME = "test_horizon_3.tsv"
STOPWORDS = stopwords.words("english")
SAVE_DIR = r"D:\data\bert_news_sentiment\reuters\vadar"


def remove_stopwords(sentence):
    return " ".join([w for w in sentence.split() if w not in STOPWORDS])


def main():
    df_test = pd.read_csv(os.path.join(
        DATA_DIR, TEST_TSV_NAME), index_col=0, sep='\t')

    sid = SentimentIntensityAnalyzer()
    y_test = df_test["Label"].values
    y_pred = []

    for sentence in tqdm(df_test["Headline"].tolist()):
        compound_score = sid.polarity_scores(sentence)["compound"]
        positive = compound_score / 2 + 0.5
        y_pred.append([1 - positive, positive])
    y_pred = np.array(y_pred)
    np.save(os.path.join(SAVE_DIR, "pred.npy"), y_pred)

    s_accuracy = pd.Series()
    for p in np.linspace(0.01, 1, 100):
        s_accuracy.set_value(1 - p, get_accuracy(y_pred, y_test, p))
    s_accuracy.to_csv(os.path.join(SAVE_DIR, "accuracy.csv"))
    fig = plt.figure()
    s_accuracy.sort_index().plot()
    # plt.show()
    fig.savefig(os.path.join(SAVE_DIR, "accuracy.png"))


if __name__ == "__main__":
    main()

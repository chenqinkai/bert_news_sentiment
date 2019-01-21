import pandas as pd
import os
import json
import datetime
import re
from multiprocessing.dummy import Pool as ThreadPool
import itertools

"""
    get news return for one stock
    we only need to load the price csv one time, which gives us a boost in performance
"""

DATA_DIR = r"D:\data\stocknet-dataset"
WORD_TO_DELETE = ["AT_USER", "URL", '-']


def load_price_data(ticker):
    path = os.path.join(DATA_DIR, "price", "raw", "%s.csv" % ticker.upper())
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df["Adj Open"] = df["Adj Close"] / df["Close"] * df["Open"]
    return df


def clean_tweet(tweet, delete_hashtag=True):
    # TODO
    # 1. will we keep retweet (marked as "rt")
    # 2. a complete list of words to delete (AT_USER, URL, etc)
    # 3. how to deal with $
    try:
        cleaned = ""
        index_to_delete = []
        for i, w in enumerate(tweet):
            if w in WORD_TO_DELETE:
                index_to_delete += [i]
                continue
            if w == '$':
                if delete_hashtag:
                    index_to_delete += [i, i + 1]
                else:
                    index_to_delete += [i]
                continue
            if not re.search(r"[a-zA-Z]", w):
                # delete word which contains only symbols
                index_to_delete += [i]
                continue
        for i in sorted(index_to_delete, reverse=True):
            del tweet[i]
    except IndexError:
        # error example:
        # ['$', 'aapl', 'commentary', 'in', 'AT_USER', 'opening', 'print', '$', '$']
        print(tweet)
    return ' '.join(tweet)


def read_json_news(ticker, date, keep_rt=True, delete_hashtag=True):
    # type(date) is str
    path = os.path.join(DATA_DIR, "tweet", "preprocessed",
                        ticker.upper(), date)
    with open(path) as f:
        tweets = []
        for line in f.readlines():
            tweets += [json.loads(line)]
    df_news = pd.DataFrame()
    for index, tweet in enumerate(tweets):
        if not keep_rt and tweet["text"][0] == 'rt':
            continue
        s = pd.Series()
        s.set_value("Ticker", ticker)
        s.set_value("DateTime", tweet["created_at"]
                    [:-10] + tweet["created_at"][-4:])
        s.set_value("Content", clean_tweet(tweet["text"], delete_hashtag))
        s.name = index
        df_news = df_news.append(s)
    return df_news


def get_all_news(ticker, save_dir, keep_rt=True, delete_hashtag=True):
    print(ticker)
    news_dir = os.path.join(DATA_DIR, "tweet", "preprocessed", ticker)
    df_all_news = pd.DataFrame()
    for f in os.listdir(news_dir):
        # if not f.startswith("2014-01"):
        #     continue
        df_news = read_json_news(
            ticker, f, keep_rt=keep_rt, delete_hashtag=delete_hashtag)
        df_all_news = pd.concat([df_all_news, df_news])
    if df_all_news.shape[0] == 0:
        print("%s does not have any entry" % ticker)
    df_all_news.index = range(df_all_news.shape[0])
    df_all_news["DateTime"] = pd.to_datetime(
        df_all_news["DateTime"], format="%a %b %d %H:%M:%S %Y").dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
    # Attention: the date has already been transformed to US/Eastern time
    df_all_news.to_csv(os.path.join(save_dir, "%s.tsv" %
                                    ticker), date_format="%Y-%m-%d %H:%M:%S", sep='\t')
    return


def main():
    save_dir = "D:/data/bert_news_sentiment/tweet"
    all_tickers = os.listdir(os.path.join(DATA_DIR, "tweet", "preprocessed"))
    pool = ThreadPool(8)
    _ = pool.starmap(get_all_news, zip(
        all_tickers, itertools.repeat(save_dir), itertools.repeat(True), itertools.repeat(False)))
    pool.close()
    pool.join()
    # get_all_news("TM", save_dir, True, False)
    return


if __name__ == "__main__":
    main()

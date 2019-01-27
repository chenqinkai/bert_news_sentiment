import datetime
import os

import pandas as pd
from pandas.tseries.offsets import BDay
from multiprocessing.dummy import Pool as ThreadPool


# TODO: add options to get return not only for 1 day, but also for multiple days
# do not work for BABA, as it starts trading on 2014-09, hence no price for 2014-07


US_MARKET_OPEN = datetime.time(hour=9, minute=30)
US_MARKET_CLOSE = datetime.time(hour=16, minute=30)
TWEET_DIR = r"D:\data\bert_news_sentiment\tweet"
PRICE_DIR = r"D:\data\stocknet-dataset\price\raw"


def load_news(ticker):
    path = os.path.join(TWEET_DIR, "%s.tsv" % ticker)
    df = pd.read_csv(path, sep="\t", index_col=0)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    return df


def load_price_data(ticker):
    path = os.path.join(PRICE_DIR, "%s.csv" % ticker.upper())
    df = pd.read_csv(path, index_col=0)
    df["Adj Open"] = df["Adj Close"] / df["Close"] * df["Open"]
    return df


def is_market_open(date_time, df_price):
    """
        0: market open
        1: market close, night
        -1: market close, morning
        2: market close, weekend
        3: market close, weekday vacation
    """
    if date_time.weekday() <= 4 and date_time.strftime("%Y-%m-%d") not in df_price.index:
        return 3
    elif date_time.weekday() <= 4 and date_time.time() >= US_MARKET_OPEN and date_time.time() <= US_MARKET_CLOSE:
        return 0
    elif date_time.weekday() <= 4 and date_time.time() > US_MARKET_CLOSE:
        return 1
    elif date_time.weekday() <= 4 and date_time.time() < US_MARKET_OPEN:
        return -1
    elif date_time.weekday() > 4:
        return 2


def get_non_holiday_start_end(start, end, df_price):
    if start.strftime("%Y-%m-%d") not in df_price.index:
        start -= BDay(1)
    if end.strftime("%Y-%m-%d") not in df_price.index:
        end += BDay(1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def get_return(df_news, df_price):
    # TODO: vectorize to be faster
    for index, row in df_news.iterrows():
        news_dt = row["DateTime"]
        df_news.loc[index, "Status"] = is_market_open(news_dt, df_price)
        if df_news.loc[index, "Status"] == 0:
            news_d_str = news_dt.strftime("%Y-%m-%d")
            df_news.loc[index, "Return"] = df_price.loc[news_d_str,
                                                        "Adj Close"] / df_price.loc[news_d_str, "Adj Open"] - 1
        elif df_news.loc[index, "Status"] == 1:
            start = news_dt
            end = news_dt + BDay(1)
            start, end = get_non_holiday_start_end(start, end, df_price)
            df_news.loc[index, "Return"] = df_price.loc[end,
                                                        "Adj Open"] / df_price.loc[start, "Adj Close"] - 1
        elif df_news.loc[index, "Status"] == -1:
            start = news_dt - BDay(1)
            end = news_dt
            start, end = get_non_holiday_start_end(start, end, df_price)
            df_news.loc[index, "Return"] = df_price.loc[end,
                                                        "Adj Open"] / df_price.loc[start, "Adj Close"] - 1
        elif df_news.loc[index, "Status"] == 2:
            start = news_dt - BDay(1)
            end = news_dt + BDay(1)
            start, end = get_non_holiday_start_end(start, end, df_price)
            df_news.loc[index, "Return"] = df_price.loc[end,
                                                        "Adj Open"] / df_price.loc[start, "Adj Close"] - 1
    return df_news


def process_one_stock(ticker):
    print(ticker)
    save_dir = "D:/data/bert_news_sentiment/tweet_with_return"
    df_news = load_news(ticker)
    df_price = load_price_data(ticker)
    df_news = get_return(df_news, df_price)
    df_news.to_csv(os.path.join(save_dir, "%s.tsv" % ticker), sep='\t')


def main():
    all_tickers = [f.rstrip(".tsv") for f in os.listdir(TWEET_DIR)]
    pool = ThreadPool(8)
    _ = pool.starmap(process_one_stock, zip(all_tickers))
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()

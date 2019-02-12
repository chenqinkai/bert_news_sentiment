import itertools
import os
from multiprocessing.dummy import Pool as ThreadPool

import pandas as pd
from pandas.tseries.offsets import BDay

PRICE_FOLDER = r"D:\data\historical_prices\full_history"
NEWS_FOLDER = r"D:\data\reuters_headlines_by_ticker\raw"

SXXP_PRICE = pd.read_csv(
    r"D:\data\historical_prices\historical_sp500.csv", index_col=0, parse_dates=True)


def load_price(ticker):
    df_price = pd.read_csv(os.path.join(
        PRICE_FOLDER, "%s.csv" % ticker), index_col=0, parse_dates=True)
    return df_price


def load_news(ticker):
    df_news = pd.read_csv(os.path.join(NEWS_FOLDER, "%s.csv" % ticker))
    df_news.columns = ["Ticker", "Name", "Date",
                       "CleanHeadline", "Headline", "Type", "Recommandation"]
    df_news["Date"] = pd.to_datetime(df_news["Date"], format="%Y%m%d")
    return df_news


def get_one_news_return(df_price, news_date, horizon=3):
    try:
        # get prev price
        prev_date = news_date - BDay(1)
        if not prev_date in df_price.index:
            prev_date -= BDay(1)
        prev_price = df_price.loc[prev_date, "adjclose"]
        prev_sp500 = SXXP_PRICE.loc[prev_date, "Adj Close"]

        # get next price
        next_date = news_date + BDay(horizon)
        if not next_date in df_price.index:
            next_date += BDay(1)
        next_price = df_price.loc[next_date, "adjclose"]
        next_sp500 = SXXP_PRICE.loc[next_date, "Adj Close"]

    except KeyError as e:
        # print("%s is not dealable" % str(news_date))
        # print(e)
        return 0
    return next_price / prev_price - next_sp500 / prev_sp500


def get_all_news_return(df_news, df_price, horizons=[1, 2, 3, 4, 5]):
    # TODO: vectorize
    for index, row in df_news.iterrows():
        for horizon in horizons:
            df_news.loc[index, "return_%d" % horizon] = get_one_news_return(
                df_price, row["Date"], horizon)
    return df_news


def main(ticker):
    print(ticker)
    df_news = load_news(ticker)
    try:
        df_price = load_price(ticker)
    except FileNotFoundError:
        print("%s does not have price information" % ticker)
        return
    df_news_with_return = get_all_news_return(df_news, df_price)
    df_news_with_return.to_csv(
        r"D:\data\reuters_headlines_by_ticker\with_return\%s.csv" % ticker, index=False)


def main_multi():
    all_ticker = os.listdir(r"D:\data\reuters_headlines_by_ticker\raw")
    all_ticker = [t.rstrip(".csv") for t in all_ticker]
    # all_ticker = all_ticker[:20]  # debug
    pool = ThreadPool(8)
    _ = pool.starmap(main, zip(all_ticker))
    pool.close()
    pool.join()


if __name__ == "__main__":
    main_multi()

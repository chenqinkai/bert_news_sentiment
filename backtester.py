import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def load_prediction(prediction_tsv_path):
    """
        load prediction into a list from tsv
    """
    with open(prediction_tsv_path, 'r') as f:
        df_pred = pd.read_csv(f, sep='\t', header=None)
    return (df_pred[1] - 0.5).tolist()


def calculate_daily_pnl(df_daily, return_col_name):
    """
        cauculate daily PnL based on return represented by return_col_name
        df_should contain a column called "Position", which indicates the nominal of stocks that we should get
    """
    return (df_daily["Position"] * df_daily[return_col_name]).sum()


def load_test_tsv(path, prediction):
    # path = r"D:\data\reuters_headlines_by_ticker\horizon_3\test_horizon_3.tsv"
    df = pd.read_csv(path, sep='\t')
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df["Prediction"] = prediction
    return df


def prediction_to_position(df_test):
    th_upper = df_test["Prediction"].quantile(0.9)
    th_lower = df_test["Prediction"].quantile(0.1)
    df_test["Position"] = 0
    df_test.loc[df_test["Prediction"] >= th_upper, "Position"] = 10000
    df_test.loc[df_test["Prediction"] <= th_lower, "Position"] = -10000
    # print(abs(df_test["Position"]).sum())
    return df_test


def get_stats(indicator_by_date):
    result = {
        "total_pnl": 0,
        "total_turnover": 0,
        "bp": 0,
        "sharpe": 0
    }
    total_pnl = indicator_by_date["pnl"].sum()
    total_turnover = indicator_by_date["turnover"].sum()
    bp = total_pnl / total_turnover
    sharpe = indicator_by_date["pnl"].mean() / indicator_by_date["pnl"].var()
    result["total_pnl"] = total_pnl
    result["total_turnover"] = total_turnover
    result["bp"] = bp
    result["sharpe"] = sharpe
    return result


def get_pnl_by_date(df_test_with_position):
    all_dates = np.unique(df_test_with_position["Date"])
    indicator_by_date = pd.DataFrame()
    for d in sorted(all_dates):
        df_daily = df_test_with_position[df_test_with_position["Date"] == d]
        pnl = calculate_daily_pnl(df_daily, "return_3")
        turnover = abs(df_daily["Position"]).sum()
        indicator_by_date.loc[d, "pnl"] = pnl
        indicator_by_date.loc[d, "turnover"] = turnover
    stats = get_stats(indicator_by_date)
    fig, ax = plt.subplots()
    ax.plot(indicator_by_date["pnl"].cumsum())
    ax.text(0.05, 0.95, str(stats), transform=ax.transAxes,
            fontsize=14, verticalalignment='top')
    plt.show()
    return indicator_by_date


def main():
    prediction = load_prediction(
        r"D:\data\reuters_headlines_by_ticker\horizon_3\bert_models_reuters_horizon_3_percentile_10_prediction_test_horizon_3_test_results.tsv")
    df_test = load_test_tsv(
        r"D:\data\reuters_headlines_by_ticker\horizon_3\test_horizon_3.tsv", prediction)
    df_test = prediction_to_position(df_test)
    pnl_by_date = get_pnl_by_date(df_test)
    print(pnl_by_date)


if __name__ == "__main__":
    main()

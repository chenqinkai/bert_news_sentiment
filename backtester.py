import argparse
import math
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.tseries.offsets import BDay
from tqdm import tqdm


def load_prediction(prediction_file_path):
    """
        load prediction into a list from tsv
    """
    if prediction_file_path.endswith(".tsv"):
        with open(prediction_file_path, 'r') as f:
            df_pred = pd.read_csv(f, sep='\t', header=None)
        return ((df_pred[1] - 0.5) * 2).tolist()
    elif prediction_file_path.endswith(".npy"):
        df_pred = np.load(prediction_file_path)
        return (df_pred[:, 1] - 0.5) * 2
    else:
        raise NotImplementedError("only .tsv and .npy are accepted")


def calculate_daily_pnl(df_daily, return_col_name):
    """
        cauculate daily PnL based on return represented by return_col_name
        df_should contain a column called "Position", which indicates the nominal of stocks that we should get
    """
    return (df_daily["Position"] * df_daily[return_col_name]).sum()


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


def prediction_to_position(df_test):
    """
        where we define our trading strategy
    """
    th_upper = df_test["Prediction"].quantile(0.99)
    th_lower = df_test["Prediction"].quantile(0.01)
    df_test["Position"] = 0
    df_test.loc[df_test["Prediction"] >= th_upper, "Position"] = 10000
    df_test.loc[df_test["Prediction"] <= th_lower, "Position"] = -10000
    # print(abs(df_test["Position"]).sum())
    return df_test


def dict_to_str(d):
    string = ""
    for k, v in d.items():
        string += '\n'
        if k == "total_pnl" or k == "total_turnover":
            string += "%s: %.2fM" % (k, v / 1000000)
        else:
            string += "%s: %.4f" % (k, v)
    return string


def get_stats(indicator_by_date):
    result = {
        "total_pnl": 0,
        "total_turnover": 0,
        "bp": 0,
        "sharpe": 0
    }
    total_pnl = indicator_by_date["pnl"].sum()
    total_turnover = indicator_by_date["turnover"].sum()
    bp = total_pnl / total_turnover * 10000
    sharpe = indicator_by_date["pnl"].mean(
    ) / indicator_by_date["pnl"].std() * math.sqrt(indicator_by_date.shape[0])
    result["total_pnl"] = total_pnl
    result["total_turnover"] = total_turnover
    result["bp"] = bp
    result["sharpe"] = sharpe
    return result


def get_pnl_by_date(df_test_with_position, save_path=None, show_fig=True):
    all_dates = np.unique(df_test_with_position["Date"])
    indicator_by_date = pd.DataFrame()
    for d in sorted(all_dates):
        df_daily = df_test_with_position[df_test_with_position["Date"] == d]
        pnl = calculate_daily_pnl(df_daily, "return_3")
        turnover = abs(df_daily["Position"]).sum() * 2
        indicator_by_date.loc[d, "pnl"] = pnl
        indicator_by_date.loc[d, "turnover"] = turnover
    stats = get_stats(indicator_by_date)
    fig, ax = plt.subplots()
    ax.plot(indicator_by_date["pnl"].cumsum())
    ax.text(0.05, 0.95, dict_to_str(stats), transform=ax.transAxes,
            fontsize=14, verticalalignment='top')
    if show_fig:
        plt.show()
    if save_path:
        fig.savefig(save_path)
    return indicator_by_date, stats


def parse_one_argument(arg):
    if '-' in arg and arg.count('-') == 1:
        return tuple(arg.split('-'))
    if '-' in arg and arg.startswith("lstm-"):
        return tuple(["shape", arg[5:]])
    if arg in ["lstm", "gru", "rnn"]:
        return tuple(["cell_type", arg])
    if arg in ["bert", "berttuned"]:
        return tuple(["tuned", arg == "berttuned"])
    raise ValueError("argument not recognized: %s" % arg)


def parse_file_argument(file_name, model_type="rnn"):
    """
        parse file names like:
        bert_label-010_emd-768_maxlen-32_lstm-128-128-128_drop-050_epoch-3_lr-100_batch-128_layer-2
        bert_label-010_emd-768_maxlen-32_lstm-128-64-32_drop-050_epoch-5/
        horizon-3_percentile-10_epoch-3_batch-32_lr-2_maxlen-32
    """
    if model_type not in ["rnn", "bert"]:
        raise ValueError("model_type should be either rnn or bert")
    file_name = file_name.split('.')[0]
    d = {
        "tuned": False,
        "label": 10,
        "emd": 768,
        "maxlen": 32,
        "shape": "128-128-128",
        "drop": 50,
        "epoch": 3,
        "lr": 100,
        "batch": 128,
        "layer": 1,
        "cell_type": "lstm",
        "full_name": file_name
    } if model_type == "rnn" else {
        "horizon": 3,
        "percentile": 10,
        "epoch": 3,
        "batch": 32,
        "lr": 2,
        "max_len": 32
    }
    splited = file_name.split('_')
    for arg in splited:
        k, v = parse_one_argument(arg)
        d[k] = v
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_path")
    parser.add_argument("--save_path")
    parser.add_argument("--prediction_dir")
    parser.add_argument(
        "--save_dir", default=r"D:\data\bert_news_sentiment\reuters\backtest\event_driven\bert")
    args = parser.parse_args()
    if (args.prediction_dir and args.prediction_path) or (not args.prediction_dir and not args.prediction_path):
        raise ValueError(
            "you should specify either a directory or a file to do backtest")
    # PREDICTION_DIR = r"D:\data\bert_news_sentiment\reuters\prediction"
    df_stat = pd.DataFrame()
    df_test = load_test_tsv(
        r"D:\data\reuters_headlines_by_ticker\horizon_3\test_horizon_3.tsv")
    is_predict_dir = args.prediction_dir is not None
    if is_predict_dir:
        files_to_backtest = os.listdir(args.prediction_dir)
    else:
        files_to_backtest = [args.prediction_path]
    for index, file_name in tqdm(enumerate(files_to_backtest)):
        prediction = load_prediction(
            os.path.join(args.prediction_dir if is_predict_dir else "", file_name))
        df_test_copy = df_test.copy()
        df_test_copy["Prediction"] = prediction
        df_test_copy = prediction_to_position(df_test_copy)
        pnl_by_date, stats = get_pnl_by_date(df_test_copy, save_path=os.path.join(
            args.save_dir if is_predict_dir else "", file_name.split('.')[0] + ".png"), show_fig=False)
        if is_predict_dir:
            param = parse_file_argument(file_name)
            param.update(stats)
            df_stat = df_stat.append(pd.Series(param, name=index))
    if is_predict_dir:
        df_stat.to_csv(os.path.join(args.save_dir, "summary.csv"))


if __name__ == "__main__":
    main()

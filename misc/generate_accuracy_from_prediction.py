import sys
import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

sys.path.append("../")
from utils import *


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_location", "-l", required=True)
    # parser.add_argument("--save_location", "-s")
    parser.add_argument("--directory", action="store_true")
    parser.add_argument("--full", action=True,
                        help="get full accuracy list on all percentiles")
    return parser


def get_one_pred_stat(y_test, y_pred, ret, percentiles_list=[1, 2, 5, 10, 100]):
    d = {}
    for p in percentiles_list:
        acc, mcc = get_accuracy(y_pred, y_test, p / 100.,
                                transform=False, calculate_mcc=True)
        d["%d_acc" % p] = acc
        d["%d_mcc" % p] = mcc
    # d["whighted_sum"] = sum(y_pred * ret)
    return d


def main():
    parser = get_parser()
    args = parser.parse_args()

    df_test = load_test_tsv(
        r"D:\data\reuters_headlines_by_ticker\horizon_3\test_horizon_3.tsv")
    y_test = (df_test["Label"].values - 0.5) * 2
    ret = df_test["return_3"].values
    if args.directory:
        df_stat = pd.DataFrame()
        for pred_name in tqdm(os.listdir(args.load_location)):
            pred_path = os.path.join(args.load_location, pred_name)
            y_pred = load_prediction(pred_path)
            stat = get_one_pred_stat(y_test, y_pred, ret)
            df_stat = df_stat.append(pd.Series(stat, name=pred_name))
        df_stat.to_csv(os.path.join(
            args.load_location, "summary_accuracy.csv"))
    else:
        y_pred = load_prediction(args.load_location)
        if args.full:
            stat = get_one_pred_stat(y_test, y_pred, ret, range(1, 101))
        else:
            stat = get_one_pred_stat(y_test, y_pred, ret)
        pd.Series(stat, name=os.path.basename(args.load_location).split(
            '.')[0]).to_csv(args.load_location.split('.')[0] + "_accuracy.csv")


if __name__ == "__main__":
    main()

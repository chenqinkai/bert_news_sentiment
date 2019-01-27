import pandas as pd
import os

TWEET_WITH_RETURN_DIR = "D:/data/bert_news_sentiment/tweet_with_return"


def cat_all_csv(file_dir, save_path):
    df_all = []
    for f in os.listdir(file_dir):
        print(f)
        df = pd.read_csv(os.path.join(file_dir, f), index_col=0, sep='\t')
        df_all += [df]
    df_all = pd.concat(df_all)
    df_all.index = range(df_all.shape[0])
    df_all.to_csv(save_path, sep='\t')
    return df_all

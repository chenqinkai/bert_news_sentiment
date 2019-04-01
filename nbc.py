import os
import pickle

import numpy as np
import pandas as pd
from nltk import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.tokenize import word_tokenize
from tqdm import tqdm

if os.name == "nt":
    DATA_DIR = r"D:\data\reuters_headlines_by_ticker\horizon_3"
elif os.name == "posix":
    DATA_DIR = "/home/chenqinkai/data/bert-news-sentiment/news/reuters/horizon_3"
TRAIN_TSV_NAME = "training_horizon_3_percentile_10.tsv"
TEST_TSV_NAME = "test_horizon_3.tsv"
STOPWORDS = stopwords.words("english")


def format_data(df):
    data = []
    for tup in zip(map(word_tokenize, df["Headline"].tolist()), df["Label"].tolist()):
        data.append(tup)
    return data


def classify_test_set(classifier, test_set):
    result = []
    for index in tqdm(range(len(test_set))):
        class_proba = classifier.prob_classify(test_set[index][0])
        result.append([class_proba.prob(0), class_proba.prob(1)])
    return np.array(result)


def main():
    df_train = pd.read_csv(os.path.join(
        DATA_DIR, TRAIN_TSV_NAME), index_col=0, sep='\t')
    train_data = format_data(df_train)
    df_test = pd.read_csv(os.path.join(
        DATA_DIR, TEST_TSV_NAME), index_col=0, sep='\t')
    test_data = format_data(df_test)
    sentim_analyzer = SentimentAnalyzer()
    all_words_neg = sentim_analyzer.all_words(
        [mark_negation(doc) for doc in train_data])
    unigram_feats = sentim_analyzer.unigram_word_feats(
        all_words_neg, min_freq=4)
    sentim_analyzer.add_feat_extractor(
        extract_unigram_feats, unigrams=unigram_feats)
    train_set = sentim_analyzer.apply_features(train_data)

    # use it only when you train the classifier for the first time
    # trainer = NaiveBayesClassifier.train
    # classifier = sentim_analyzer.train(trainer, train_set)

    if os.name == "nt":
        classifier_path = r'D:\data\bert_news_sentiment\reuters\nbc\nbc_percent-10.pickle'
    elif os.name == "posix":
        classifier_path = "/home/chenqinkai/data/bert-news-sentiment/nbc/nbc_percent-10.pickle"
    with open(classifier_path, 'rb') as handle:
        classifier = pickle.load(handle)

    test_set = sentim_analyzer.apply_features(test_data)
    result = classify_test_set(classifier, test_set)


if __name__ == "__main__":
    main()

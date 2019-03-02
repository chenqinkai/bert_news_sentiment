import numpy as np
np.random.seed(0)
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
# to make sure that the generated model is the same for each time
np.random.seed(1)
from generate_embedding import generate_embedding_to_file
import os
import pandas as pd
from matplotlib import pyplot as plt

MAX_LEN = 10
EMBEDDING_SIZE = 768


def clean_one_sentence(text, word_to_index, no_stopword=False):
    # TODO: decide if we should clean headlines before generating embedding
    text_clean = text.lower()
    text_clean = re.sub(r"[^a-zA-Z]", " ", text_clean)
    if no_stopword:
        all_words = [w for w in text_clean.split(
        ) if w not in STOPWORDS and w in word_to_index]
    else:
        all_words = [w for w in text_clean.split() if w in word_to_index]
    return " ".join(all_words)


def convert_to_one_hot(Y):
    '''
        Y: np.array
        return: np.array
    '''
    Y_oh = np.zeros((Y.shape[0], 2))
    Y_oh[:, 0] = 1 - Y
    Y_oh[:, 1] = Y
    return Y_oh


def get_accuracy(y_pred, y_test, percentile=1.0):
    df = pd.DataFrame()
    df['pred'] = (y_pred[:, 1] - 0.5) * 2
    df['test'] = (y_test - 0.5) * 2
    upper = df['pred'].quantile(1 - percentile / 2.)
    lower = df['pred'].quantile(percentile / 2.)
    df = df[(df['pred'] >= upper) | (df['pred'] <= lower)]
    correct = df[np.sign(df['pred']) == np.sign(df['test'])]
    return correct.shape[0] / float(df.shape[0])


def get_model(input_dim=EMBEDDING_SIZE, input_length=MAX_LEN):
    print("Start Initialzing Neural Network!")
    model = Sequential()
    model.add(LSTM(64, input_dim=input_dim, input_length=input_length,
                   return_sequences=True, activation='tanh'))
    model.add(Dropout(0.5))
    # output shape : (4,4)
    model.add(LSTM(32, return_sequences=False, activation='tanh'))
    model.add(Dropout(0.5))
    # output shape : (4,)
    model.add(Dense(2, activation='softmax'))
    # output shape : (3,)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    model = get_model()
    print(model.summary())

    data_dir = "D:/data/reuters_headlines_by_ticker/horizon_3"
    train_tsv_name = "training_horizon_3_percentile_10.tsv"
    test_tsv_name = "test_horizon_3.tsv"

    df_train = pd.read_csv(os.path.join(
        data_dir, train_tsv_name), index_col=0, sep='\t')
    df_test = pd.read_csv(os.path.join(
        data_dir, test_tsv_name), index_col=0, sep='\t')

    train_embedding_path = os.path.join(
        data_dir, "embeddings", train_tsv_name[:-4] + "_base.npy")
    test_embedding_path = os.path.join(
        data_dir, "embeddings", test_tsv_name[:-4] + "_base.npy")

    if not os.path.exists(train_embedding_path):
        print("generating embedding for training set")
        generate_embedding_to_file(
            df_train["Headline"].tolist(), MAX_LEN, train_embedding_path)
    X_train = np.load(train_embedding_path)
    Y_train = df_train['Label'].values
    Y_train_oh = convert_to_one_hot(Y_train)

    if not os.path.exists(test_embedding_path):
        print("generating embedding for test set")
        generate_embedding_to_file(
            df_test["Headline"].tolist(), MAX_LEN, test_embedding_path)
    X_test = np.load(test_embedding_path)
    Y_test = df_test['Label'].values

    # launch training, time consuming!!!
    # verbose=2 to avoid progress bar multi-line printing problem
    model.fit(x=X_train, y=Y_train_oh, epochs=5, batch_size=32, verbose=2)
    model_dir = r"D:\data\bert_news_sentiment\reuters\model\bert_label-010_emd-%d_maxlen-%d_lstm-64-32_drop-050_epoch-5" % (
        EMBEDDING_SIZE, MAX_LEN)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(os.path.join(model_dir, "model.h5"))

    # out-of-sample prediction
    y_pred = model.predict(X_test)
    pd.DataFrame(y_pred).to_csv(os.path.join(
        model_dir, "result.csv"), index=False, header=False)

    # to plot and save plot
    s_accuracy = pd.Series()
    for p in np.linspace(0.01, 1, 100):
        s_accuracy.set_value(1 - p, get_accuracy(y_pred, Y_test, p))
    s_accuracy.to_csv(os.path.join(model_dir, "accuracy.csv"))
    fig = plt.figure()
    s_accuracy.sort_index().plot()
    fig.savefig(os.path.join(model_dir, "accuracy.png"))

    # X = np.array([np.random.rand(10, 768) for x in range(100)])
    # Y_oh = np.zeros((100, 2))
    # Y_oh[:, 1] = 1 - Y_oh[:, 0]

    # model.fit(x=X, y=Y_oh,
    #           epochs=5, batch_size=10, verbose=1)

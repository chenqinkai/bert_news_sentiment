import numpy as np
np.random.seed(0)
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
# to make sure that the generated model is the same for each time
np.random.seed(1)
import os
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf

MAX_LEN = 10  # @param {type:"integer"}
EMBEDDING_SIZE = 768  # @param {type:"integer"}
WE_ARE_ON_GCP = False  # @param {type:"boolean"}

if not WE_ARE_ON_GCP:
    from generate_embedding import generate_embedding_to_file
else:
    from google.colab import auth
    auth.authenticate_user()


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


def get_model(input_shape=(MAX_LEN, EMBEDDING_SIZE)):
    print("Start Initialzing Neural Network!")
    model = Sequential()
    model.add(LSTM(64, return_sequences=True,
                   activation="tanh", input_shape=input_shape))
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

    if WE_ARE_ON_GCP:
        data_dir = "gs://bert-news-sentiment/reuters/horizon_3"
    else:
        data_dir = "D:/data/reuters_headlines_by_ticker/horizon_3"
    train_tsv_name = "training_horizon_3_percentile_10.tsv"
    test_tsv_name = "test_horizon_3.tsv"

    with tf.gfile.Open(os.path.join(data_dir, train_tsv_name), 'r') as f_train:
        df_train = pd.read_csv(f_train, index_col=0, sep='\t')
    with tf.gfile.Open(os.path.join(data_dir, test_tsv_name), 'r') as f_test:
        df_test = pd.read_csv(f_test, index_col=0, sep='\t')

    if WE_ARE_ON_GCP:
        train_embedding_path = "gs://bert-news-sentiment/rnn/embedding/training_horizon_3_percentile_10_base_layer_1_maxlen_20.npy"
        test_embedding_path = "gs://bert-news-sentiment/rnn/embedding/test_horizon_3_base_layer_1_maxlen_20.npy"
    else:
        # if we are running locally, check if embedding exists
        train_embedding_path = os.path.join(
            data_dir, "embeddings", train_tsv_name[:-4] + "_base.npy")
        test_embedding_path = os.path.join(
            data_dir, "embeddings", test_tsv_name[:-4] + "_base.npy")
        if not os.path.exists(train_embedding_path):
            print("generating embedding for training set")
            generate_embedding_to_file(
                df_train["Headline"].tolist(), MAX_LEN, train_embedding_path)
        if not os.path.exists(test_embedding_path):
            print("generating embedding for test set")
            generate_embedding_to_file(
                df_test["Headline"].tolist(), MAX_LEN, test_embedding_path)

    with tf.gfile.Open(train_embedding_path, 'rb') as f_train_emb:
        X_train = np.load(f_train_emb)
    Y_train = df_train['Label'].values
    Y_train_oh = convert_to_one_hot(Y_train)

    with tf.gfile.Open(test_embedding_path, 'rb') as f_test_emb:
        X_test = np.load(f_test_emb)
    Y_test = df_test['Label'].values

    # launch training, time consuming!!!
    # verbose=2 to avoid progress bar multi-line printing problem
    model.fit(x=X_train, y=Y_train_oh, epochs=5, batch_size=32, verbose=2)

    # save model
    if WE_ARE_ON_GCP:
        model_dir = "gs://bert-news-sentiment/rnn/model/bert_label-010_emd-%d_maxlen-%d_lstm-64-32_drop-050_epoch-5" % (
            EMBEDDING_SIZE, MAX_LEN)
    else:
        model_dir = r"D:\data\bert_news_sentiment\reuters\model\bert_label-010_emd-%d_maxlen-%d_lstm-64-32_drop-050_epoch-5" % (
            EMBEDDING_SIZE, MAX_LEN)
    if not tf.gfile.Exists(model_dir):
        tf.gfile.MakeDirs(model_dir)
    with tf.gfile.Open(os.path.join(model_dir, "model.h5"), 'w') as f_model:
        model.save(f_model)

    # out-of-sample prediction
    y_pred = model.predict(X_test)
    with tf.gfile.Open(os.path.join(model_dir, "result.csv"), 'w') as f_result:
        pd.DataFrame(y_pred).to_csv(f_result, index=False, header=False)

    # to plot and save plot
    s_accuracy = pd.Series()
    for p in np.linspace(0.01, 1, 100):
        s_accuracy.set_value(1 - p, get_accuracy(y_pred, Y_test, p))
    with tf.gfile.Open(os.path.join(model_dir, "accuracy.csv"), 'w') as f_accuracy:
        s_accuracy.to_csv(f_accuracy)
    fig = plt.figure()
    s_accuracy.sort_index().plot()
    with tf.gfile.Open(os.path.join(model_dir, "accuracy.png"), 'w') as f_accuracy_plot:
        fig.savefig(f_accuracy_plot)

    # X = np.array([np.random.rand(10, 768) for x in range(100)])
    # Y_oh = np.zeros((100, 2))
    # Y_oh[:, 1] = 1 - Y_oh[:, 0]

    # model.fit(x=X, y=Y_oh,
    #           epochs=5, batch_size=10, verbose=1)

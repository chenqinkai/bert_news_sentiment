import numpy as np
np.random.seed(0)
import pandas as pd
import bs4
import re
from keras.layers import Embedding, Input, LSTM, Dropout, Dense, Activation
from keras.models import Model
from nltk.corpus import stopwords
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
# to make sure that the generated model is the same for each time
np.random.seed(1)

from utils import get_accuracy

EMBEDDING_SIZE = 100
MAX_LEN = 10
NO_STOPWORD = True
STOPWORDS = set(stopwords.words('english') +
                ["rt", " "])  # remove "rt": retweet


def clean_one_review(text, word_to_index, no_stopword=False):
    # because in movie reviews, sometimes there are html tags
    bs = bs4.BeautifulSoup(text, "lxml")
    text_clean = bs.text.lower()
    text_clean = re.sub(r"[^a-zA-Z]", " ", text_clean)
    if no_stopword:
        all_words = [w for w in text_clean.split(
        ) if w not in STOPWORDS and w in word_to_index]
    else:
        all_words = [w for w in text_clean.split() if w in word_to_index]
    return " ".join(all_words)


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_len = len(word_to_index) + 1

    ### START CODE HERE ###
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, EMBEDDING_SIZE))

    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False.
    embedding_layer = Embedding(vocab_len, EMBEDDING_SIZE, trainable=False)
    ### END CODE HERE ###

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def readGloveFile(gloveFile):
    # Prepare Glove File
    with open(gloveFile, 'r') as f:
        wordToGlove = {}  # map from a token (word) to a Glove embedding vector
        wordToIndex = {}  # map from a token to an index
        indexToWord = {}  # map from an index to a token

        for line in f:
            record = line.strip().split()
            token = record[0]  # take the token (word) from the text line
            # associate the Glove embedding vector to a that token (word)
            wordToGlove[token] = np.array(record[1:], dtype=np.float64)

        tokens = sorted(wordToGlove.keys())
        for idx, tok in enumerate(tokens):
            # 0 is reserved for masking in Keras (see above)
            kerasIdx = idx + 1
            wordToIndex[tok] = kerasIdx  # associate an index to a token (word)
            # associate a word to a token (word). Note: inverse of dictionary above
            indexToWord[kerasIdx] = tok

    return wordToIndex, indexToWord, wordToGlove


def sentence_to_index(X, word_to_index):
    """
        X: list
        return: np.array(len(X), MAX_LEN)
    """
    X_index = np.zeros((len(X), MAX_LEN))
    for i in range(len(X)):
        words = X[i].split()
        if len(words) > MAX_LEN:
            words = words[:MAX_LEN]
        for j, w in enumerate(words):
            X_index[i, j] = word_to_index[w]
    return X_index


def get_model(input_shape, word_to_glove, word_to_index):
    one_sentence_index = Input(shape=input_shape, dtype='int32')
    embedding_layer = pretrained_embedding_layer(word_to_glove, word_to_index)
    embedding = embedding_layer(one_sentence_index)

    X = LSTM(64, return_sequences=True)(embedding)
    X = Dropout(0.5)(X)
    X = LSTM(32, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(2)(X)
    X = Activation('softmax')(X)

    model = Model(inputs=one_sentence_index, outputs=X)
    return model


def convert_to_one_hot(Y):
    '''
        Y: np.array
        return: np.array
    '''
    Y_oh = np.zeros((Y.shape[0], 2))
    Y_oh[:, 0] = 1 - Y
    Y_oh[:, 1] = Y
    return Y_oh


def label_training_data(df_train, percent):
    """
        label top <percent> data as positive
        label end <percent> data as negative
    """
    df_train["DateTime"] = pd.to_datetime(df_train["DateTime"])
    selected_data_number = int(df_train.shape[0] * percent)
    df_positive = df_train.sort_values("Return")[:selected_data_number]
    df_positive["Label"] = 1
    df_negative = df_train.sort_values(
        "Return")[df_train.shape[0] - selected_data_number:]
    df_negative["Label"] = 0
    return pd.concat([df_positive, df_negative]).sort_values("DateTime")


def main():
    # before launching, set gpu_options.allow_growth to True to avoid memory errors
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    # dynamically grow the memory used on the GPU
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # set this TensorFlow session as the default session for Keras
    set_session(sess)

    # load data
    word_to_index, index_to_word, word_to_glove = readGloveFile(
        "D:/data/nlp/glove/glove.6B.%dd.txt" % (EMBEDDING_SIZE))
    df_train = pd.read_csv(
        r"D:\data\reuters_headlines_by_ticker\horizon_3\training_horizon_3_percentile_10.tsv", index_col=0, sep='\t')
    df_test = pd.read_csv(
        r"D:\data\reuters_headlines_by_ticker\horizon_3\test_horizon_3.tsv", index_col=0, sep='\t')
    # df_train = label_training_data(df_train, 0.1)
    # df_test = label_training_data(df_test, 0.5)

    # clean data
    print("cleaning train data ...")
    X_train = [clean_one_review(str(review), word_to_index, NO_STOPWORD)
               for review in tqdm(df_train['Headline'].values)]
    Y_train = df_train['Label'].values
    print("cleaning test data ...")
    X_test = [clean_one_review(str(review), word_to_index, NO_STOPWORD)
              for review in tqdm(df_test['Headline'].values)]
    Y_test = df_test['Label'].values

    # convert to required format
    X_train_index = sentence_to_index(X_train, word_to_index)
    X_test_index = sentence_to_index(X_test, word_to_index)
    Y_train_oh = convert_to_one_hot(Y_train)

    # get model
    model = get_model((MAX_LEN,), word_to_glove, word_to_index)
    print(model.summary())
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # launch training, time consuming!!!
    # verbose=2 to avoid progress bar multi-line printing problem
    model.fit(x=X_train_index, y=Y_train_oh,
              epochs=5, batch_size=32, verbose=2)
    model_dir = r"D:\data\bert_news_sentiment\reuters\model\w2v_label-010_emd-%d_maxlen-%d_lstm-64-32_drop-05_epoch-5" % (
        EMBEDDING_SIZE, MAX_LEN)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(os.path.join(model_dir, "model.h5"))

    # out-of-sample prediction
    y_pred = model.predict(X_test_index)
    pd.DataFrame(y_pred).to_csv(os.path.join(
        model_dir, "result.csv"), index=False, header=False)

    # to plot and save plot
    s_accuracy = pd.Series()
    for p in np.linspace(0.01, 1, 100):
        s_accuracy.set_value(1 - p, get_accuracy(y_pred, Y_test, p))
    s_accuracy.to_csv(os.path.join(model_dir, "accuracy.csv"))
    fig = plt.figure()
    s_accuracy.sort_index().plot()
    # plt.show()
    fig.savefig(os.path.join(model_dir, "accuracy.png"))


# when running with unenough memory on GPU, run this command to force the use of CPU (for Windows)
# set CUDA_VISIBLE_DEVICES=-1
if __name__ == '__main__':
    main()

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
# to make sure that the generated model is the same for each time
np.random.seed(1)

EMBEDDING_SIZE = 100
MAX_LEN = 10
NO_STOPWORD = True
STOPWORDS = set(stopwords.words('english') + ["rt"])  # remove "rt": retweet


def clean_one_review(text, word_to_index, no_stopword=False):
    bs = bs4.BeautifulSoup(text)
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


def get_accuracy(y_pred, y_test, percentile=1.0):
    df = pd.DataFrame()
    df['pred'] = (y_pred[:, 1] - 0.5) * 2
    df['test'] = (y_test - 0.5) * 2
    upper = df['pred'].quantile(1 - percentile / 2.)
    lower = df['pred'].quantile(percentile / 2.)
    df = df[(df['pred'] >= upper) | (df['pred'] <= lower)]
    correct = df[np.sign(df['pred']) == np.sign(df['test'])]
    return correct.shape[0] / float(df.shape[0])


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
    # load data
    word_to_index, index_to_word, word_to_glove = readGloveFile(
        "D:/data/nlp/glove/glove.6B.%dd.txt" % (EMBEDDING_SIZE))
    df_train = pd.read_csv(
        "D:/data/bert_news_sentiment/train.tsv", index_col=0, sep='\t')
    df_test = pd.read_csv(
        "D:/data/bert_news_sentiment/test.tsv", index_col=0, sep='\t')
    df_train = label_training_data(df_train, 0.1)
    df_test = label_training_data(df_train, 0.5)

    # clean data
    X_train = [clean_one_review(str(review), word_to_index, NO_STOPWORD)
               for review in tqdm(df_train['Content'].values)]
    Y_train = df_train['Label'].values
    X_test = [clean_one_review(str(review), word_to_index, NO_STOPWORD)
              for review in df_test['Content'].values]
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
    model.fit(x=X_train_index, y=Y_train_oh, epochs=10, batch_size=32)
    model.save(r"D:\data\bert_news_sentiment\model\rnn\model.h5")

    # out-of-sample prediction
    y_pred = model.predict(X_test_index)
    # accuray_oos = get_accuracy(y_pred, Y_test, percentile=0.5)

    # to plot and save plot
    s_accuracy = pd.Series()
    for p in np.linspace(0, 1, 101):
        s_accuracy.set_value(1 - p, get_accuracy(y_pred, Y_test, p))
    fig = plt.figure()
    s_accuracy.sort_index().plot()
    # plt.show()
    fig.savefig(r"D:\data\bert_news_sentiment\model\rnn\model.png")


if __name__ == '__main__':
    main()

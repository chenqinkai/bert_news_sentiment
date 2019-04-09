import tensorflow as tf
import numpy as np
import os
import argparse
from keras.models import load_model
from tensorflow.python.lib.io import file_io
from tqdm import tqdm


ALL_MODEL_DIR = "gs://bert-news-sentiment/rnn/model/reuters"
EMBEDDING_DIR = "gs://bert-news-sentiment/rnn/embedding"


def get_embedding_param(dir_name):
    max_len = None
    for param_str in dir_name.split('_'):
        if param_str.startswith("maxlen"):
            max_len = int(param_str.split('-')[1])
            break
    return max_len, "tuned" in dir_name


def get_all_model_params(all_model_dir):
    result = {}
    all_models = tf.gfile.ListDirectory(all_model_dir)
    for m in all_models:
        m_param = get_embedding_param(m)
        if m_param not in result:
            result[m_param] = [m]
        else:
            result[m_param] += [m]
    return result


def download_model_file(model_dir):
    model_file = file_io.FileIO(os.path.join(
        ALL_MODEL_DIR, model_dir, "model.h5"), mode='rb')
    temp_model_location = './temp_model.h5'
    temp_model_file = open(temp_model_location, 'wb')
    temp_model_file.write(model_file.read())
    temp_model_file.close()
    model_file.close()
    return


def get_embedding_name(maxlen, tuned=False, part=None):
    return "test_horizon_3_%s_layer_%d_maxlen_%d%s.npy" % (
        "tuned" if tuned else "base",
        1 if tuned else 2, maxlen,
        "_part%d" % part if part is not None else "")


def main():
    all_model_params = get_all_model_params(ALL_MODEL_DIR)

    for k, v in all_model_params.items():
        if k[0] != 64 or k[1] != False:
            continue
        print("we are dealing with '%s'" % str(k))
        if k[0] >= 64:
            result = {}
            embedding_name = get_embedding_name(k[0], k[1], part=1)
            print("loading part 1")
            with tf.gfile.Open(os.path.join(EMBEDDING_DIR, embedding_name), 'rb') as f_emb_1:
                X_test_1 = np.load(f_emb_1)
            print("predicting part 1")
            for model_dir in tqdm(v):
                download_model_file(model_dir)
                model = load_model("./temp_model.h5")
                y_pred = model.predict(X_test_1)
                result[model_dir] = y_pred
            del X_test_1

            embedding_name = get_embedding_name(k[0], k[1], part=2)
            print("loading part 2")
            with tf.gfile.Open(os.path.join(EMBEDDING_DIR, embedding_name), 'rb') as f_emb_2:
                X_test_2 = np.load(f_emb_2)
            print("predicting part 2")
            for model_dir in tqdm(v):
                download_model_file(model_dir)
                model = load_model("./temp_model.h5")
                y_pred = model.predict(X_test_2)
                result[model_dir] = np.append(
                    result[model_dir], y_pred, axis=0)
            del X_test_2

            for k, v in result.items():
                with tf.gfile.Open(os.path.join(ALL_MODEL_DIR, k, "pred.npy"), 'w') as f_pred:
                    np.save(f_pred, v)
                with tf.gfile.Open(os.path.join("gs://bert-news-sentiment/rnn", "%s.npy" % k.rstrip('/')), 'w') as f_pred:
                    np.save(f_pred, v)
        else:
            embedding_name = get_embedding_name(k[0], k[1], part=None)
            print("loading")
            with tf.gfile.Open(os.path.join(EMBEDDING_DIR, embedding_name), 'rb') as f_emb:
                X_test = np.load(f_emb)
            print("predicting")
            for model_dir in tqdm(v):
                download_model_file(model_dir)
                model = load_model("./temp_model.h5")
                y_pred = model.predict(X_test)
                with tf.gfile.Open(os.path.join(ALL_MODEL_DIR, model_dir, "pred.npy"), 'w') as f_pred:
                    np.save(f_pred, y_pred)
                with tf.gfile.Open(os.path.join("gs://bert-news-sentiment/rnn/prediction", "%s.npy" % model_dir.rstrip('/')), 'w') as f_pred:
                    np.save(f_pred, y_pred)
            del X_test


if __name__ == "__main__":
    main()

import numpy as np
import tensorflow as tf
from gensim.models import fasttext, word2vec

import config


def load_data():
    """
    加载处理好的数据
    :return: 数据
    """
    train_X = np.load(config.train_x_path + '.npy')
    train_Y = np.load(config.train_y_path + '.npy')
    test_X = np.load(config.test_x_path + '.npy')
    return train_X, train_Y, test_X


def load_embedding_matrix():
    return np.load(config.embedding_matrix_path)


def load_model(method):
    if method == "fasttext":
        model = fasttext.FastText.load(config.model_path)
    else:
        model = word2vec.Word2Vec.load(config.model_path)
    embedding_matrix = np.load(config.embedding_matrix_path)
    vocab_i2w = {idx: word for idx, word in enumerate(model.wv.index2word)}
    vocab_w2i = {word: idx for idx, word in enumerate(model.wv.index2word)}
    return model, embedding_matrix, vocab_i2w, vocab_w2i


def config_gpu():
    """
    配置GPU环境
    :return:
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

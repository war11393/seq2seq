import re
import jieba
import logging
import numpy as np
import pandas as pd
from gensim.models import fasttext
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence

import config
from utils.multi_proc_util import parallelize

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 加载自定义词典
jieba.load_userdict(config.user_dict)
# 加载停用词
stopwords = open(config.stopword_list, 'r', encoding='UTF-8').read().split('\n')


def pre_processing(params):
    # 读取数据集
    df_train = pd.read_csv(config.train_set, encoding='UTF-8')
    df_test = pd.read_csv(config.test_set, encoding='UTF-8')

    # 丢弃训练集结果栏含空值的行，为其余空值补空串
    df_train.dropna(subset=['Report'], inplace=True)
    df_train.fillna('', inplace=True)
    df_test.fillna('', inplace=True)

    # 多核分词操作
    df_train = parallelize(df_train, proc_df)
    df_test = parallelize(df_test, proc_df)

    # 合并相关数据
    df_train['merged'] = df_train[['Question', 'Dialogue', 'Report']].apply(lambda x: ' '.join(x), axis=1)
    df_test['merged'] = df_test[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    df_merged = pd.concat([df_train[['merged']], df_test[['merged']]], axis=0)
    print('训练集: {},测试集: {},合并后: {}'.format(len(df_train), len(df_test), len(df_merged)))

    # 保存总词表
    df_merged.to_csv(config.words_file, index=None, header=False)

    # 训练词向量
    print('开始训练向量模型')
    vector_train = fasttext_proc if params['vector_train_method'] == "fasttext" else word2vec_proc
    model = vector_train(params)

    vocab = model.wv.vocab

    # 取出训练用数据
    df_train['X'] = df_train[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    df_train['Y'] = df_train['Report']
    df_test['X'] = df_test[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)

    # 求取最大长度  --  此处改为手动指定长度
    # max_x1 = get_max_len(df_train['X'])
    # max_x2 = get_max_len(df_test['X'])
    # params.max_x_length = max(max_x1, max_x2)
    # params.max_y_length = get_max_len(df_train['Y'])

    # 对文本进行填充处理
    df_train['X'] = df_train['X'].apply(lambda x: pad_text(x, params['max_x_length'], vocab, False))
    df_test['X'] = df_test['X'].apply(lambda x: pad_text(x, params['max_x_length'], vocab, False))
    df_train['Y'] = df_train['Report'].apply(lambda x: pad_text(x, params['max_y_length'], vocab, True))

    # 保存填充结果,便于查看
    df_train['X'].to_csv(config.train_context, index=None, header=False)
    df_train['Y'].to_csv(config.train_report, index=None, header=False)
    df_test['X'].to_csv(config.test_context, index=None, header=False)

    # 再次训练词向量
    print('再次训练词向量')
    model.build_vocab(LineSentence(config.train_context), update=True)
    model.train(LineSentence(config.train_context), epochs=3, total_examples=model.corpus_count)
    model.build_vocab(LineSentence(config.train_report), update=True)
    model.train(LineSentence(config.train_report), epochs=3, total_examples=model.corpus_count)
    model.build_vocab(LineSentence(config.test_context), update=True)
    model.train(LineSentence(config.test_context), epochs=3, total_examples=model.corpus_count)

    print('重训练后词汇表总长度为 {} \n保存模型数据 ...'.format(len(model.wv.vocab)))

    # 保存模型
    model.save(config.model_path)

    # 对x、y的值转码成数字序列，用于模型训练，保存训练数据
    vocab_w2i = {word: index for index, word in enumerate(model.wv.index2word)}

    train_ids_x = df_train['X'].apply(lambda x: transform_data(x, vocab_w2i))
    train_X = np.array(train_ids_x.tolist())
    np.save(config.train_x_path, train_X)

    train_ids_y = df_train['Y'].apply(lambda x: transform_data(x, vocab_w2i))
    train_Y = np.array(train_ids_y.tolist())
    np.save(config.train_y_path, train_Y)

    test_ids_x = df_test['X'].apply(lambda x: transform_data(x, vocab_w2i))
    test_X = np.array(test_ids_x.tolist())
    np.save(config.test_x_path, test_X)
    print('模型训练结束')


def transform_data(text, vocab_w2i):
    """
    词转索引
    :param text: 词序列 [word1,word2,word3, ...]
    :param vocab_w2i: 词汇表 此时表中已有<unk> 和 <pad>, 可不需判断词是否在词表中
    :return: 索引序列 [index1,index2,index3 ......]
    """
    words = text.split(' ')
    ids = [vocab_w2i[w] if w in vocab_w2i else vocab_w2i['<unk>'] for w in words]
    return ids


def pad_text(text, max_len, vocab, flag):
    """
    自动处理、填充文本
    :param text: 文本
    :param max_len: 最大长度
    :param vocab: 词典
    :param flag: 是否填充<start><end>
    :return: 填充后文本
    """
    words = text.strip().split(' ')
    words = words[:max_len]
    text = [w if w in vocab else '<unk>' for w in words]
    if flag:
        text = ['<start>'] + words + ['<end>']
    else:
        text
    text = text + ['<pad>'] * (max_len - len(words))
    return ' '.join(text)


def proc_text(text):
    """
    处理文本：清理文本，切词，去停用，拼接
    :param text:待处理文本
    :return: 处理后文本
    """
    text = clean_text(text)
    tokens = jieba.cut(text)
    tokens = [w for w in tokens if w not in stopwords]  # 去停
    return ' '.join(tokens)


def proc_df(df):
    """
    适配处理两类数据集进行处理，忽略'Brand', 'Model'两列
    :param df: 数据集
    :return: 处理后数据集
    """
    for col in ['Question', 'Dialogue']:
        df[col] = df[col].apply(proc_text)
    if 'Report' in df.columns:
        df['Report'] = df['Report'].apply(proc_text)
    return df


def clean_text(text):
    """
    将文本中的指定字符移除，保留一部分标点
    :param text: 待处理文本
    :return: 处理后文本
    """
    if isinstance(text, str):
        return re.sub(
            r'[\s+\-\!\/\|\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?！？【】“”~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好，|您好，|你好！|您好！',
            '', text)
    else:
        return ' '


# 获取最大长度
def get_max_len(words):
    """
    求得所有句子的平均长度
    :param words:
    :return:
    """
    lengths = words.apply(lambda x: x.count(' ') + 1)
    return int(np.mean(lengths) + 2 * np.std(lengths))


# 保存至文件
def save_words(words, path):
    of = open(path, 'w', encoding='UTF-8')
    for idx, w in enumerate(words):
        of.write(str(w) + '\n')
    of.close()


# 词向量训练方法
def word2vec_proc(params):
    line_sentence = LineSentence(config.words_file)
    model = word2vec.Word2Vec(line_sentence, size=params['vector_dim'], window=params['window_size'],
                              min_count=params['min_frequency'], workers=params['workers'], sg=params['use_skip_gram'],
                              hs=params['use_hierarchical_softmax'], negative=params['negative_size'],
                              iter=params['pre_proc_epochs'])
    return model


def fasttext_proc(params):
    line_sentence = LineSentence(config.words_file)
    model = fasttext.FastText(line_sentence, size=params['vector_dim'], window=params['window_size'],
                              min_count=params['min_frequency'], workers=params['workers'], sg=params['use_skip_gram'],
                              hs=params['use_hierarchical_softmax'], negative=params['negative_size'],
                              iter=params['pre_proc_epochs'])
    return model


if __name__ == '__main__':
    pre_processing(config.get_params())

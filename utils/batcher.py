import tensorflow as tf

from utils.utils import load_data


def train_batch_generator(batch_size, sample_sum=None):
    # 加载数据集
    train_X, train_Y, _ = load_data()
    if sample_sum:
        train_X = train_Y[:sample_sum]
        train_Y = train_Y[:sample_sum]
    dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).shuffle(len(train_X))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    total_steps = len(train_X) // batch_size
    return dataset, total_steps


def test_batch_generator(batch_size):
    # 加载数据集
    _, _, test_X = load_data()
    dataset = tf.data.Dataset.from_tensor_slices(test_X)
    dataset = dataset.batch(batch_size)
    total_steps = len(test_X) // batch_size
    if len(test_X) % batch_size != 0:
        total_steps += 1
    return dataset, total_steps

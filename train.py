import time
import tensorflow as tf

import config
from models import Seq2Seq
from utils.utils import load_model, config_gpu
from utils.batcher import train_batch_generator


# Homework-week4
# 利用前三节作业内容，搭建自己的seq2seq模型，并用项目数据模型训练。
#
# 1. 根据week3完成情况最终搭建好seq2seq模型
# 模型中加载week2预训练词向量
#
# 2. 调整week1作业vocab，并建立导入模型的数据
# 将start等特殊符号考虑其中 将预处理后的训练数据导入到模型中
#
# 3. 完成loss函数的构建，设置优化器等
# 4. 完成基本模型seq2seq的训练
def train_model(model, params, ckpt):
    _, embedding_matrix, vocab_i2w, vocab_w2i = load_model(params['vector_train_method'])

    batch_size = params['batch_size']
    pad_index = vocab_w2i['<pad>']
    unk_index = vocab_w2i['<unk>']

    # 优化器
    optimizer = tf.keras.optimizers.Adagrad(params['learning_rate'],
                                            initial_accumulator_value=params['adagrad_init_acc'],
                                            clipnorm=params['max_grad_norm'])

    # 交叉熵计算
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(real, pred):
        pad_mask = tf.math.equal(real, pad_index)
        nuk_mask = tf.math.equal(real, unk_index)
        mask = tf.math.logical_not(tf.math.logical_or(pad_mask, nuk_mask))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    @tf.function
    def batch_train(enc_input, report):
        """
        批训练
        :param enc_input: 一个batch的x
        :param report: 一个batch的y
        :return:
        """
        with tf.GradientTape() as gt:
            enc_output, enc_hidden = model.call_encoder(enc_input)  # 进行编码
            dec_hidden = enc_hidden  # 第一个解码器隐层输入
            dec_input = tf.expand_dims([vocab_w2i['<start>']] * batch_size, 1)
            dec_output, dec_hidden = model.call_decoder(dec_input, dec_hidden, enc_output, report)
            loss = loss_function(report[:, 1:], dec_output)
            # 反向梯度求导
            variables = model.encoder.trainable_variables + model.decoder.trainable_variables + model.attention.trainable_variables
            gradients = gt.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            return loss

    dataset, steps_per_epoch = train_batch_generator(batch_size)
    for epoch in range(params['train_epochs']):
        print('阶段 {} 开始训练'.format(epoch + 1))
        start = time.time()
        total_loss = 0
        for (batch, (batch_x, batch_y)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = batch_train(batch_x, batch_y)
            total_loss += batch_loss
            if batch % 100 == 0:
                print('阶段 {} 第 {} 批数据完成， 总Loss {:.4f}'.format(epoch + 1, batch + 1, batch_loss))
        print('阶段 {} / {} 完成, 总Loss {:.4f}'.format(epoch + 1, params['train_epochs'], total_loss / steps_per_epoch))
        print('此阶段耗时 {} 秒\n'.format(time.time() - start))
        ckpt.save()
        print('纪录checkpoint...\n')


def train(params):
    config_gpu()  # 配置GPU环境

    # 构建模型
    print("创建模型 ...")
    model = Seq2Seq(params=params)

    # 获取保存管理者
    print("创建模型保存器")
    checkpoint = tf.train.Checkpoint(Seq2Seq=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, config.checkpoint_path, max_to_keep=3)
    if checkpoint_manager.latest_checkpoint:
        print("加载最新保存器数据 {} ...".format(checkpoint_manager.latest_checkpoint))
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
    else:
        print("初始化保存器.")
    # 训练模型
    print("开始训练 ...")
    train_model(model, params, checkpoint_manager)


if __name__ == '__main__':
    train(config.get_params())

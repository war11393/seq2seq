import tensorflow as tf
import pandas as pd
import time
from tqdm import tqdm

import config
from models import Seq2Seq
from utils.batcher import test_batch_generator
from utils.utils import load_model, config_gpu


# Homework-week5
# 经历前四节作业后，已经可以完成基本的seq2seq模型训练，本次作业最终目标是通过训练结果进行测试。
# 1. 理解Beam Search核心代码的结构
# 2. 将Beam Search代码融合到测试阶段的Decoder里，完成最终测试代码
def inference_model(model, params):
    _, embedding_matrix, vocab_i2w, vocab_w2i = load_model(params['vector_train_method'])

    def batch_inference(inp):
        """
        批预测
        :param inp: 输入
        :return:
        """
        params['batch_size'] = len(inp)
        encoder_output, encoder_hidden = model.call_encoder(inp)
        decoder_hidden = encoder_hidden
        dec_input = tf.expand_dims([vocab_w2i['<start>']] * params['batch_size'], 1)

        predicts = [''] * params['batch_size']
        for idx in range(params['max_y_length']):
            context_vector, _ = model.attention(decoder_hidden, encoder_output)
            prediction, decoder_hidden = model.decoder(dec_input, decoder_hidden, encoder_output, context_vector)

            predicted_id = tf.argmax(prediction, axis=1).numpy()

            for i, v in enumerate(predicted_id):
                predicts[i] += vocab_i2w[v] + ' '

            # teach forcing
            dec_input = tf.expand_dims(predicted_id, 1)
        return predicts

    def result_proc(text):
        """
        对预测结果做最后处理
        :param text: 单条预测结果
        :return:
        """
        # text = text.lstrip(' ，！。')
        text = text.replace(' ', '')
        text = text.strip()
        if '<end>' in text:
            text = text[:text.index('<end>')]
        return text

    print('开始预测...')
    start = time.time()
    results = []
    dataset, steps_per_epoch = test_batch_generator(params['batch_size'])
    with tqdm(total=steps_per_epoch, position=0, leave=True) as tq:
        for (batch, batch_x) in enumerate(dataset.take(steps_per_epoch)):
            results += batch_inference(batch_x)
            tq.update(1)

    print('预测完成，耗时{}s\n处理至文件...'.format(time.time() - start))

    test_csv = pd.read_csv(config.test_set, encoding="UTF-8")
    # 赋值结果
    test_csv['Prediction'] = results
    # 提取ID和预测结果两列
    test_df = test_csv[['QID', 'Prediction']]
    # 结果处理
    test_df['Prediction'] = test_df['Prediction'].apply(result_proc)
    # 保存结果
    test_df.to_csv(config.inference_result_path, index=None, sep=',')
    print('已保存文件至{}'.format(config.inference_result_path))


def inference(params):
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
    # 训练模型
    print("开始预测 ...")
    inference_model(model, params)


if __name__ == '__main__':
    params = config.get_params()
    params['batch_size'] = 256
    inference(params)

import os
import pathlib
import argparse

# 获取项目根目录
root = pathlib.Path(os.path.abspath(__file__)).parent

# 数据集文件路径
train_set = os.path.join(root, 'dataset', 'AutoMaster_TrainSet.csv')
test_set = os.path.join(root, 'dataset', 'AutoMaster_TestSet.csv')

# 文件分词处理后保存路径
train_context = os.path.join(root, 'vocabs', 'train_context.txt')
test_context = os.path.join(root, 'vocabs', 'test_context.txt')
train_report = os.path.join(root, 'vocabs', 'train_report.txt')

# 文件分词处理结果转化为索引后保存路径
train_x_path = os.path.join(root, 'vocabs', 'train_x_idx')
train_y_path = os.path.join(root, 'vocabs', 'train_y_idx')
test_x_path = os.path.join(root, 'vocabs', 'test_x_idx')

# 词表存储路径
words_file = os.path.join(root, 'vocabs', 'words.txt')

# 词向量预训练模型保存路径
model_path = os.path.join(root, 'vocabs', 'vector.model')
embedding_matrix_path = os.path.join(root, 'vocabs', 'vector.model.wv.vectors.npy')

# 训练模型保存路径
checkpoint_path = os.path.join(root, 'training_checkpoints')

# 其他文件路径
stopword_list = os.path.join(root, 'other', 'stopword_list')
user_dict = os.path.join(root, 'other', 'user_dict')

# 预测结果输出路径
inference_result_path = os.path.join(root, 'output', 'inference_result.csv')


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='模型执行模式', type=str)
    # Pre Processing
    parser.add_argument('--vector_dim', default=256, help='训练过程中参与训练的词向量维度大小', type=int)
    parser.add_argument('--max_x_length', default=128, help='正文文本分词后内容的最大长度', type=int)
    parser.add_argument('--max_y_length', default=34, help='摘要信息内容的最大长度', type=int)
    parser.add_argument('--min_y_length', default=2, help='摘要信息内容的最大长度', type=int)
    parser.add_argument('--vector_train_method', default='word2vec', help='训练词向量的方法，默认word2vec,可选fasttext', type=str)
    parser.add_argument('--pre_proc_epochs', default=6, help='设置进行处理的迭代次数', type=int)
    parser.add_argument('--window_size', default=5, help='进行上下文选取时的窗口大小', type=int)
    parser.add_argument('--min_frequency', default=3, help='最低词频', type=int)
    parser.add_argument('--workers', default=10, help='用多少个内核来进行处理', type=int)
    parser.add_argument('--use_skip_gram', default=1, help='是否使用skip-gram，否则CBOW', type=int)
    parser.add_argument('--use_hierarchical_softmax', default=0, help='是否使用多层softmax，否则negative(负采样)', type=int)
    parser.add_argument('--negative_size', default=5, help='若采用负采样，设置负样本数', type=int)
    parser.add_argument('--vocab_size', default=45354, help='完成训练后词表长度', type=int)
    # train-inference
    parser.add_argument('--batch_size', default=128, help='一批训练多少个样本', type=int)
    parser.add_argument('--encoder_units', default=128, help='编码器隐层单元数', type=int)
    parser.add_argument('--decoder_units', default=128, help='解码器隐层单元数', type=int)
    parser.add_argument('--attn_units', default=128, help='注意力层隐层单元数', type=int)
    parser.add_argument('--train_epochs', default=1, help='训练次数', type=int)
    parser.add_argument('--beam_size', default=2, help='集束搜索的集束宽度', type=int)
    parser.add_argument('--learning_rate', default=0.001, help='集束搜索的集束宽度', type=float)
    parser.add_argument("--adagrad_init_acc", default=0.1, help="Adagrad优化器初始累加器值", type=float)
    parser.add_argument("--max_grad_norm", default=0.8, help="剪裁渐变的渐变范数", type=float)

    return vars(parser.parse_args())

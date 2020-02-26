# v1.0 baseline版本
可正常运行的baseline，数据集使用百度AIStudio提供的汽车大师问答摘要与推理项目数据集，[点此查看](https://aistudio.baidu.com/aistudio/datasetdetail/1407)

baseline基于tensorflow-2.0进行搭建，相关信息如下：
- 模型可选用gensim中的word2vec或fasttext完成词向量的预训练，并在之后的训练中不再参与训练;
- encoder-decoder使用的基本单元为单向gru
- decoder采用了Bahdanau的Attention计算方法进行优化
- loss使用tf2中提供的tf.keras.losses.SparseCategoricalCrossentropy
- optimizer使用tf.keras.optimizers.Adagrad
- 推理方法使用贪心搜索

将训练结果提交比赛平台后分数为21.4607

# v1.1 调整推理方式-beam search
把贪心换成集束搜索，推理耗时相当大。

将训练结果提交比赛平台后分数为22.7544
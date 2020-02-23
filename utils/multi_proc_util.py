from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd


def parallelize(df, func):
    """
    多核并行处理模块
    :param df: DataFrame数据
    :param func: 预处理函数
    :return: 处理后的数据
    """
    # cpu 数量、分块个数
    cores = cpu_count()
    print('{}核处理本次任务'.format(cores))
    # 数据切分
    data_split = np.array_split(df, cores)
    # 线程池
    pool = Pool(cores)
    # 数据分发 合并
    data = pd.concat(pool.map(func, data_split))
    # 关闭线程池
    pool.close()
    # 执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    pool.join()
    print('任务处理完毕')
    return data

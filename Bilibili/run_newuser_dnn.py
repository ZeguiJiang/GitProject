# coding=utf-8
# /usr/bin/evn python

'''
Author: Yuan Fei Xiaoming
Email: arccos2002@gmail.com
Date: 2019-08-16 15:42
Desc:
'''
import sys
import argparse
import tensorflow as tf
from os.path import join
from smart_io.readConf import resolveJson


from model.dnn import dnn
from smart_io.read_data import read_dataset
from util.parseFeature import parser_feature
from util.feature import *
from util.utils import *
from util.format_weight import format_weight
#from util import export_model
import glob
import pandas as pd
import configparser
import collections


def is_done(file_path):
    n = 0
    while n < 20:
        if tf.io.gfile.exists(file_path):
            return True
        print("ROUND %d,Data NOT Ready for Training: %s" % (n, file_path))
        time.sleep(45 * 60)  # 等待45分钟
        n = n + 1
    return tf.io.gfile.exists(file_path)


def file_lists(pre_fix):
    return pre_fix  + "/*"


def test_lists(pre_fix):
    return pre_fix  + "/*"


def auc(labels, predictions):
    return tf.metrics.auc(labels=labels,
                          predictions=predictions[:, 1],
                          name='auc_op')


def my_auc(labels, predictions):

    labels = tf.reshape(labels,[-1,1])
    logistic = tf.cast(predictions['logistic'],tf.int32)
    logistic = predictions['logistic']
    auc_metric = tf.compat.v1.metrics.auc(labels,logistic)
    tf.summary.scalar('my_auc',auc_metric[1])
    return {'my_auc': auc_metric}


if __name__ == "__main__":

    date0 = (datetime.now() - timedelta(1)).strftime('%Y%m%d')  # 昨天


    # 读取配置文件
    config = "conf/conf.json"
    configs = resolveJson(config)

    column_dict = configs['COLUMN_DICT']

    configMap = configs['TASK_CONFIG']
    data = read(date0,configMap)
    print(data)


    column_dict = column_dict_preprocess(column_dict)# 这个col dic 是什么


    use_col, feature_cols, features = gen_feat(column_dict)





    done_path = data['FEAT_TRAIN'] + ".done" # done 是什么　

    if not is_done(done_path):
       print("Data NOT Ready for Training: " + done_path)
       sys.exit()
    
    # 读取训练数据集
    train_files = tf.io.gfile.glob(file_lists(data['FEAT_TRAIN']))
    # 读取测试数据集
    test_files = tf.io.gfile.glob(test_lists(data['FEAT_TEST']))

    estimator = dnn(feature_cols, configMap)
    decode_col, decode_col_idx,default_type = genRecord_defaults(column_dict)

    train_spec = tf.estimator.TrainSpec(
        input_fn=read_dataset(train_files, configMap, decode_col, default_type,decode_col_idx, tf.estimator.ModeKeys.TRAIN),
        max_steps=configMap['max_step'])
    eval_spec = tf.estimator.EvalSpec(
        input_fn=read_dataset(test_files, configMap, decode_col, default_type, decode_col_idx,tf.estimator.ModeKeys.EVAL),
        steps=configMap['val_step'], throttle_secs=8, start_delay_secs=5)
    train_and_eval = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    #w = format_weight(estimator, True, known_args, list(use_col.keys()))


    export_model.export_model(estimator,use_col,configMap)





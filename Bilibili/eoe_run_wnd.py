
'''
Author: shuwei Huang
email: x1ao.shu@163.com
Date: 20220916
'''

from model.widendeep import wdl
from util.utils import *
from smart_io.readConf import *
import sys
from smart_io.readConf import resolveJson
from model.dnn import dnn
from smart_io.read_data import read_dataset
from util.parseFeature import parser_feature
from util.format_weight import format_weight

from util.feature import *
from util.utils import *
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

def getFileList(path):
   return [path + x for x in os.listdir(path) if not x.startswith('.')]

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

    #mode = "local"
    mode = "remote"

    date0 = (datetime.now() - timedelta(1)).strftime('%Y%m%d')  # 昨天
    #date0 = "20211123"
    print(date0)

    if mode == "local":
        config = "conf/conf_wnd_local.json"
    elif mode == "remote":
        config = "conf/conf_wnd.json"
        # config = "conf/conf_wnd_cross.json"
    else:
        raise Exception("please check mode, Either 'remote' or 'local' ")


    configs = resolveJson(config)
    column_dict = configs['COLUMN_DICT']

    configMap = configs['TASK_CONFIG']

    data = read(date0, configMap)

    column_dict = column_dict_preprocess(column_dict)

    if config == "conf/conf_wnd_cross.json":
        featcol, linear_featcols, dnn_featcols = gen_feat_wnd(column_dict, make_cross_features=False)
    else:
        featcol, linear_featcols, dnn_featcols = gen_feat_wnd(column_dict, make_cross_features=True)

    # 本地读取数据
    if mode == "local":

        train_files = getFileList("data/eoe_train/train/")

        test_files = getFileList("data/eoe_test/test/")

        print("train_files:",train_files)
        print("test_files:", test_files)
    elif mode == "remote":
        # 读取训练数据集
        train_files = tf.io.gfile.glob(file_lists(data['FEAT_TRAIN']))
        # 读取测试数据集
        test_files = tf.io.gfile.glob(test_lists(data['FEAT_TEST']))
    else:
        raise Exception("please check mode, Either 'remote' or 'local' ")

    estimator = wdl(linear_featcols,dnn_featcols,configMap)

    decode_col, decode_col_idx, default_type = genRecord_defaults(column_dict)

    # print("decode", decode_col)
    # print(len(decode_col))
    # print("decode_col_idx", decode_col_idx)
    # print("default_type", default_type)




    train_spec = tf.estimator.TrainSpec(
        input_fn=read_dataset(train_files, configMap, decode_col, default_type, decode_col_idx,
                              tf.estimator.ModeKeys.TRAIN),
        max_steps=configMap['max_step'])
    eval_spec = tf.estimator.EvalSpec(
        input_fn=read_dataset(test_files, configMap, decode_col, default_type, decode_col_idx,
                              tf.estimator.ModeKeys.EVAL),
        steps=configMap['val_step'], throttle_secs=8, start_delay_secs=5)
    train_and_eval = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


    if mode == "local":
        pass
    elif mode == "remote":
        from util import export_model
        export_model.export_model(estimator, featcol, configMap)
    else:
        raise Exception("please check mode, Either 'remote' or 'local' ")



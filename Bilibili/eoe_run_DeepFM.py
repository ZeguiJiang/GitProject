
'''
Author: shuwei Huang ref：zegui
email: x1ao.shu@163.com
Date: 20220916
'''

from util.utils import *
from smart_io.readConf import *
import sys
from smart_io.readConf import resolveJson
from model.dnn import dnn
from model.DeepFM import *
from smart_io.read_data import read_dataset,read_dataset_1
from util.parseFeature import parser_feature
from util.format_weight import format_weight

from util.feature import *
from util.feature_map import *
from util.utils import *
from util import export_model

import glob
import pandas as pd
import configparser
import collections

def get_primary_key_1(train_files,   default_type):

    def decode_csv(value_column):
        value_column = tf.strings.regex_replace(value_column, r"\\N", '0')
        values = tf.io.decode_csv(value_column, record_defaults=default_type, field_delim="\001")
        mid = values[0:1]
        manga_id = values[-1]
        print('mid',mid)
        return mid,manga_id


    # 单机训练数据，如果内存装的下，可以用cache，数据太大的话，去掉cache；worker内存分配数 cpu数*3
    # dataset = tf.data.TextLineDataset(train_files).batch(
    #     known_args.val_batch_size).map(decode_csv, num_parallel_calls=os.cpu_count()).cache()
    dataset = tf.data.TextLineDataset(train_files).map(decode_csv, num_parallel_calls=os.cpu_count())
    return dataset


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

def predict_lists(pre_fix):
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

    mode = "local"
    #mode = "remote"

    date0 = (datetime.now() - timedelta(1)).strftime('%Y%m%d')  # 昨天

    #设定任务时间
    #date0 = '20211205'

    print(date0)

    #Exception("please check mode, Either 'remote' or 'local' ")

    #读取config map

    if mode == "local":
        config = "conf/conf_dfm_local.json"
    elif mode == "remote":
        config = "conf/conf_dnn.json"
        # config = "conf/conf_dnn_cross.json"
    else:
        raise Exception("please check mode, Either 'remote' or 'local' ")

    configs = resolveJson(config)
    column_dict = configs['COLUMN_DICT']
    configMap = configs['TASK_CONFIG']
    data = read(date0, configMap)
    column_dict = column_dict_preprocess(column_dict)
    use_col, feature_cols, features = gen_feat(column_dict)
    #use_col, feature_cols = gen_feat_dnn(configMap['feature_map'])

    # 本地读取数据
    if mode == "local":
        train_files = getFileList("data/train/")
        test_files = getFileList("data/test/")
        #check_files = getFileList("data/prediction/")
    elif mode == "remote":
        # 读取训练数据集
        train_files = tf.io.gfile.glob(file_lists(data['FEAT_TRAIN']))
        # 读取测试数据集
        test_files = tf.io.gfile.glob(test_lists(data['FEAT_TEST']))

        #check_files = tf.io.gfile.glob(predict_lists(data['FEAT_PREDICT']))
    else:
        raise Exception("please check mode, Either 'remote' or 'local' ")

    decode_col, decode_col_idx, default_type = genRecord_defaults(column_dict)

    estimator = deepfm(feature_cols, configMap )





    train_spec = tf.estimator.TrainSpec(
        input_fn=read_dataset(train_files, configMap, decode_col, default_type, decode_col_idx,
                              tf.estimator.ModeKeys.TRAIN),
        max_steps=configMap['max_step'])
    eval_spec = tf.estimator.EvalSpec(
        input_fn=read_dataset(test_files, configMap, decode_col, default_type, decode_col_idx,
                              tf.estimator.ModeKeys.EVAL),
        steps=configMap['val_step'], throttle_secs=8, start_delay_secs=5)
    train_and_eval = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
    #     print('Parsing', filenames)
    #
    #     def decode_libsvm(line):
    #         # columns = tf.decode_csv(value, record_defaults=CSV_COLUMN_DEFAULTS)
    #         # features = dict(zip(CSV_COLUMNS, columns))
    #         # labels = features.pop(LABEL_COLUMN)
    #         columns = tf.string_split([line], ' ')
    #         labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
    #         splits = tf.string_split(columns.values[1:], ':')
    #         id_vals = tf.reshape(splits.values, splits.dense_shape)
    #         feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2, axis=1)
    #         feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
    #         feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
    #         return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels
    #
    #     dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=10).prefetch(500000)
    #     if perform_shuffle:
    #         dataset = dataset.shuffle(buffer_size=256)
    #
    #     # epochs from blending together.
    #     dataset = dataset.repeat(num_epochs)
    #     dataset = dataset.batch(batch_size)  # Batch size to use
    #     iterator = dataset.make_one_shot_iterator()
    #     batch_features, batch_labels = iterator.get_next()
    #     # return tf.reshape(batch_ids,shape=[-1,field_size]), tf.reshape(batch_vals,shape=[-1,field_size]), batch_labels
    #     return batch_features, batch_labels
    #
    #
    # task_type = 'train'
    # if task_type == 'train':
    #     train_spec = tf.estimator.TrainSpec(
    #         input_fn=lambda: input_fn(train_files, num_epochs=1))
    #     eval_spec = tf.estimator.EvalSpec(
    #         input_fn=lambda: input_fn(test_files, num_epochs=1), steps=None,
    #         start_delay_secs=1000, throttle_secs=1200)
    #     tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)







    #
    #predict = estimator.predict(input_fn=read_dataset(test_files, configMap, decode_col, default_type, decode_col_idx, tf.estimator.ModeKeys.PREDICT))
    #labels =
    # dic_a = {}
    # dic_b = {}
    # dic_c = {}
    # dic_d = {}
    # for p in predict:
    #     print(p)
    #     if p['mid'] == b'1194020618':
    #         dic_a[p['comic_id']] = p['probabilities'][1]
    #     elif p['mid'] == b'329289277':
    #         dic_b[p['comic_id']] = p['probabilities'][1]
    #     elif p['mid'] == b'1629097540':
    #         dic_c[p['comic_id']] = p['probabilities'][1]
    #     elif p['mid'] == b'388157476':
    #         dic_d[p['comic_id']] = p['probabilities'][1]
    #
    #
    # sort_dic_a  = sorted(dic_a.items(), key=lambda x:x[1], reverse=True)
    #
    # print("user_1194020618")
    # for i, j in enumerate(sort_dic_a):
    #     if i < 20:
    #         print(j)
    #     else:
    #         break
    #
    # sort_dic_b = sorted(dic_b.items(), key=lambda x: x[1], reverse=True)
    # print("user_329289277")
    # for i, j in enumerate(sort_dic_b):
    #     if i < 20:
    #         print(j)
    #     else:
    #         break
    #
    # sort_dic_c = sorted(dic_c.items(), key=lambda x: x[1], reverse=True)
    # print("user_1629097540")
    # for i, j in enumerate(sort_dic_c):
    #     if i < 20:
    #         print(j)
    #     else:
    #         break
    #
    # sort_dic_d = sorted(dic_d.items(), key=lambda x: x[1], reverse=True)
    # print("user_388157476")
    # for i, j in enumerate(sort_dic_d):
    #     if i < 20:
    #         print(j)
    #     else:
    #         break






    # if mode == "local":
    #     from util import export_model
    #     export_model.export_model(estimator, use_col, configMap)
    # elif mode == "remote":
    #     from util import export_model
    #     export_model.export_model(estimator, use_col, configMap)
    # else:
    #     raise Exception("please check mode, Either 'remote' or 'local' ")



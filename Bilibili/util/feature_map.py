# coding=utf-8
# /usr/bin/evn python

'''
Author: zegui jiang
Email: jiangzegui@bilibili.com
Date: 2021-12-28
Desc:
'''

import tensorflow as tf
import json
import re
import os
from collections import OrderedDict
from tensorflow.python.framework import dtypes

from smart_io.readConf import resolveJson
from util.utils import column_dict_preprocess


def gen_feat_dnn(path):
    feature_cols = []
    feats = []
    use_col = {}
    cate_cols = []


    with open(path, 'r') as load_f:
        features = json.load(load_f)['features']

    for feat in features:
        if feat['feature_name'] in ['mid', 'label']:
            continue

        elif feat['feature_type'] == "combo_feature":
            feature_cols.append(tf.feature_column.indicator_column(
                tf.feature_column.crossed_column(
                    keys=[i.split(':')[1] for i in feat['expression']], hash_bucket_size=feat["hash_bucket_size"])))

        elif feat['feature_type'] == 'onehot_feature':
            feature_cols.append(tf.feature_column.numeric_column(key=feat['feature_name']))


        elif feat['value_type'] == 'int':
            fc = tf.feature_column.numeric_column(key=feat['feature_name'], dtype=tf.int32)
            feature_cols.append(fc)
            use_col[feat['feature_name']] = feat['value_type']

        elif feat['value_type'] == 'bigint':
            fc = tf.feature_column.numeric_column(key=feat['feature_name'], dtype=tf.int64)
            feature_cols.append(fc)
            use_col[feat['feature_name']] = feat['value_type']

        elif feat['value_type'] == 'float' :
            fc = tf.feature_column.numeric_column(key=feat['feature_name'])
            feature_cols.append(fc)
            use_col[feat['feature_name']] = feat['value_type']

        elif feat['value_type'] == 'double':
            fc = tf.feature_column.numeric_column(key=feat['feature_name'], dtype=dtypes.float64)
            feature_cols.append(fc)
            use_col[feat['feature_name']] = feat['value_type']

        elif feat['value_type'] == 'string':
            fc = tf.feature_column.categorical_column_with_hash_bucket(key=feat['feature_name'],hash_bucket_size=feat["hash_bucket_size"])
            feature_cols.append(tf.feature_column.indicator_column(fc))
            use_col[feat['feature_name']] = feat['value_type']
        else:
            Exception("check data type!")


    return use_col,feature_cols
import tensorflow as tf
import numpy as numpy

from util.utils import *
from smart_io.readConf import *
import sys
from smart_io.readConf import resolveJson
from model.dnn import dnn
from smart_io.read_data import read_dataset
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


if __name__ == '__main__':
    def file_lists(pre_fix):
        return pre_fix + "/*"


    def getFileList(path):
        return [path + x for x in os.listdir(path) if not x.startswith('.')]

    def decode_libsvm(line):

        # columns = tf.decode_csv(value, record_defaults=CSV_COLUMN_DEFAULTS)
        # features = dict(zip(CSV_COLUMNS, columns))
        # labels = features.pop(LABEL_COLUMN)
        columns = tf.string_split([line], ' ')
        labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
        splits = tf.string_split(columns.values[1:], ':')
        id_vals = tf.reshape(splits.values, splits.dense_shape)
        feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2, axis=1)
        feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
        feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
        return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels

    # feat_ids = [1,2,4,5,6,7]
    # field_size = 0
    # feat_ids = tf.reshape(feat_ids, shape=[-1,6])

    train_files = getFileList("/Users/monarch/PycharmProjects/model_zoo/data/train/")
    a = tf.data.TextLineDataset(train_files).map(decode_libsvm,num_parallel_calls=10)

    # print((feat_ids))

    x = tf.Variable(tf.random.uniform([5, 30], -1, 1))

    s0, s1, s2 = tf.split(x, num_or_size_splits=3, axis=1)

    print(tf.shape(s0))

    print(tf.shape(s1))
    print(tf.shape(s2).numpy())

    print(x)

    "sss".__contains__("aa")
# coding=utf-8
# /usr/bin/evn python

'''
Author: Fei Xiaoming
Email: arccos2002@gmail.com
Date: 2019-08-16 14:58
Desc: 
'''
import tensorflow as tf
from util.process import feature_preprocess
import os


def read_dataset(train_files, configMap, decode_col, default_type,decode_col_idx, mode, process=True):



    # @tf.function
    def _input_fn():
        def decode_csv(value_column):
            value_column = tf.strings.regex_replace(value_column, r"\\N", '0')
            values = tf.io.decode_csv(value_column, record_defaults=default_type, field_delim="\001", select_cols=decode_col_idx)
            features = dict(zip(decode_col, values))
            label = features.pop('label')
            if process:
                features = feature_preprocess(features)

            return features, label


        # 单机训练数据，如果内存装的下，可以用cache，数据太大的话，去掉cache；worker内存分配数 cpu数*3
        # if mode == tf.estimator.ModeKeys.TRAIN:
        #     dataset = tf.data.TextLineDataset(train_files).shuffle(configMap['buffer_size']).batch(
        #         configMap['batch_size']).map(decode_csv, num_parallel_calls=os.cpu_count()).cache().repeat(
        #         configMap['repeat'])
        # else:
        #     dataset = tf.data.TextLineDataset(train_files).batch(
        #         configMap['val_batch_size']).map(decode_csv, num_parallel_calls=os.cpu_count()).cache()
        # return dataset.prefetch(configMap['prefetch_size']).make_one_shot_iterator().get_next()

        if mode == tf.estimator.ModeKeys.TRAIN:

            dataset = tf.data.TextLineDataset(train_files).shuffle(configMap['buffer_size']).batch(
                configMap['batch_size']).map(decode_csv, num_parallel_calls=os.cpu_count()).repeat(
                configMap['repeat'])

        else:
            dataset = tf.data.TextLineDataset(train_files).batch(
                configMap['val_batch_size']).map(decode_csv, num_parallel_calls=os.cpu_count())
        return dataset.prefetch(configMap['prefetch_size']).make_one_shot_iterator().get_next()

    return _input_fn





def read_dataset_1(train_files, configMap, decode_col, default_type,decode_col_idx, mode, process=True):



    # @tf.function
    def _input_fn():
        def decode_csv(value_column):
            value_column = tf.strings.regex_replace(value_column, r"\\N", '0')
            values = tf.io.decode_csv(value_column, record_defaults=default_type, field_delim="\001", select_cols=decode_col_idx)
            features = dict(zip(decode_col, values))
            label = features.pop('label')
            if process:
                features = feature_preprocess(features)

            return features, label


        def decode_libsvm(line):
            # columns = tf.decode_csv(value, record_defaults=CSV_COLUMN_DEFAULTS)
            # features = dict(zip(CSV_COLUMNS, columns))
            # labels = features.pop(LABEL_COLUMN)
            columns = tf.string_split([line], ' ')

            labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
            print("label",labels)
            splits = tf.string_split(columns.values[1:], ':')
            print("splits", splits)
            id_vals = tf.reshape(splits.values, splits.dense_shape)

            feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2, axis=1)

            feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
            feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
            return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels



        if mode == tf.estimator.ModeKeys.TRAIN:

            # dataset = tf.data.TextLineDataset(train_files).shuffle(configMap['buffer_size']).batch(
            #     configMap['batch_size']).map(decode_csv, num_parallel_calls=os.cpu_count()).repeat(
            #     configMap['repeat'])
            dataset = tf.data.TextLineDataset(train_files).map(decode_libsvm, num_parallel_calls=10)

        else:
            dataset = tf.data.TextLineDataset(train_files).batch(
                configMap['val_batch_size']).map(decode_csv, num_parallel_calls=os.cpu_count())

        batch_features, batch_labels =  dataset.prefetch(configMap['prefetch_size']).make_one_shot_iterator().get_next()
        return batch_features, batch_labels

    return _input_fn


def get_primary_key(train_files, known_args, col_map, default_type):
    cols = col_map.keys()

    def decode_csv(value_column):
        value_column = tf.strings.regex_replace(value_column, r"\\N", '0')
        values = tf.io.decode_csv(value_column, record_defaults=default_type, field_delim="\001")
        mid = values[0:1]
        manga_id = values[-1]
        return mid,manga_id


    # 单机训练数据，如果内存装的下，可以用cache，数据太大的话，去掉cache；worker内存分配数 cpu数*3
    # dataset = tf.data.TextLineDataset(train_files).batch(
    #     known_args.val_batch_size).map(decode_csv, num_parallel_calls=os.cpu_count()).cache()
    dataset = tf.data.TextLineDataset(train_files).batch(
        known_args.val_batch_size).map(decode_csv, num_parallel_calls=os.cpu_count())
    return dataset

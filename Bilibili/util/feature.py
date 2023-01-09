# coding=utf-8
# /usr/bin/evn python

'''
Author: Yuan
Email: arccos2002@gmail.com
Date: 2019-08-13 18:24
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


def gen_feat(column_dict,make_cross_features=True):
    feat_cols = []
    feats = []
    use_col = {}
    cate_cols = []

    feature_columns = OrderedDict()

    features = dict(column_dict['FEAT_TRAIN'])

    #print("features",features)
    for k in  column_dict['FEAT_EXCLUDE']:
        print("column_dict['FEAT_EXCLUDE']", features[k])
        del features[k]
    #print("features", features)
    del features['label']  # pop label

    for feat in features:

        # if feat in ['mid']:
        #     continue

        tp =  features[feat]

        big_bucket = ['comic_id', 'manga_id','mid']

        if tp =='int':
            fc = tf.feature_column.numeric_column(feat, dtype=tf.int32)
            feat_cols.append(fc)
            feature_columns[feat] = fc
            use_col[feat] = tp

        elif tp == 'bigint':
            fc = tf.feature_column.numeric_column(feat, dtype=tf.int64)
            feat_cols.append(fc)
            feature_columns[feat] = fc
            use_col[feat] = tp

        elif tp == 'float' :
            fc = tf.feature_column.numeric_column(feat)
            feat_cols.append(fc)
            feature_columns[feat] = fc
            use_col[feat] = tp

        elif tp == 'double':
            fc = tf.feature_column.numeric_column(feat, dtype=dtypes.float64)
            feat_cols.append(fc)
            feature_columns[feat] = fc
            use_col[feat] = tp

        elif tp == 'string' and (feat in big_bucket):
            fc = tf.feature_column.categorical_column_with_hash_bucket(key=feat,hash_bucket_size=20000)
            cate_cols.append(feat)
            feat_cols.append(tf.feature_column.indicator_column(fc))
            feature_columns[feat] = fc
            use_col[feat] = tp
        else:
            fc = tf.feature_column.categorical_column_with_hash_bucket(key=feat, hash_bucket_size=200)
            cate_cols.append(feat)
            feat_cols.append(tf.feature_column.indicator_column(fc))
            feature_columns[feat] = fc
            use_col[feat] = tp
    # print("feat_cols = ", feat_cols)
    # print("feats = ", feats)
    # print("use_col =" , use_col)
    # print("cate_cols =", cate_cols)


    if make_cross_features:
        for k1, k2s in column_dict['FEAT_CROSS'].items():
            #print("k1",k1)
            #print("k2s", k2s)
            for k2 in k2s:
                k = '%s_%s' % (k1,k2)
                if k1 == 'mid' or k1 == 'manga_id':
                    fc = tf.feature_column.crossed_column([k1,k2], hash_bucket_size=20000)
                else:
                    fc = tf.feature_column.crossed_column([k1,k2],hash_bucket_size=200)
                feature_columns[k] = fc
                cate_cols.append(k)
                feat_cols.append(tf.feature_column.indicator_column(fc))






    return use_col, feat_cols, feats




def gen_feat_wnd(column_dict,make_cross_features=True):
    featcol = {}
    linear_featcols = []
    dnn_featcols = []
    use_col = {}

    features = dict(column_dict['FEAT_TRAIN'])

    print("features",features)
    for k in  column_dict['FEAT_EXCLUDE']:
        #print("column_dict[ 'FEAT_EXCLUDE']", features[k])
        del features[k]
    #print("features", features)
    del features['label']

    for feat in features:
        # print("feat = ", feat)
        # if feat in ['mid']:
        #     continue
        tp =  features[feat]
        #big_bucket = ['u_active_city_level','u_active_province','u_active_city','u_brand','u_chid_all','comic_id']
        big_bucket = ['comic_id', 'manga_id', 'mid']

        cross_bucket = ["cross_mid_comic_id","cross_mid_item_is_finish","cross_mid_item_language","cross_mid_item_main_style_id" ,"cross_u_exp_level_u_active_city_level","cross_u_exp_level_u_vip_status" ,"cross_u_exp_level_u_predict_sex" ,"cross_u_exp_level_u_predict_age_range","cross_u_exp_level_item_language","cross_u_exp_level_item_area_id","cross_u_exp_level_item_is_finish","cross_u_exp_level_item_main_style_id"]

        if tp ==  'int' :
            fc = tf.feature_column.numeric_column(feat,  dtype=tf.int32)
            linear_featcols.append(fc)
            dnn_featcols.append(fc)
            featcol[feat] = tp

        elif tp == 'bigint':
            fc = tf.feature_column.numeric_column(feat,  dtype=tf.int64)
            linear_featcols.append(fc)
            dnn_featcols.append(fc)
            featcol[feat] = tp

        elif tp == 'float':
            fc = tf.feature_column.numeric_column(feat)
            linear_featcols.append(fc)
            dnn_featcols.append(fc)
            featcol[feat] = tp

        elif tp == 'double':
            fc = tf.feature_column.numeric_column(feat,  dtype=dtypes.float64)
            linear_featcols.append(fc)
            dnn_featcols.append(fc)
            featcol[feat] = tp

        elif tp == 'string' and (feat in big_bucket):
            fc = tf.feature_column.categorical_column_with_hash_bucket(key=feat,hash_bucket_size=2000)
            linear_featcols.append(tf.feature_column.indicator_column(fc))
            #dnn_featcols.append(tf.feature_column.indicator_column(fc))
            featcol[feat] = tp

        elif tp == 'string' and (feat in cross_bucket):
            fc = tf.feature_column.categorical_column_with_hash_bucket(key=feat,hash_bucket_size=200)
            linear_featcols.append(tf.feature_column.indicator_column(fc))
            #dnn_featcols.append(tf.feature_column.indicator_column(fc))
            featcol[feat] = tp

        else:
            fc = tf.feature_column.categorical_column_with_hash_bucket(key=feat, hash_bucket_size=200)
            linear_featcols.append(tf.feature_column.indicator_column(fc))
            dnn_featcols.append(tf.feature_column.embedding_column(fc,dimension=32))
            featcol[feat] = tp



    if make_cross_features:
        for k1, k2s in column_dict['FEAT_CROSS'].items():
            for k2 in k2s:
                k = '%s_%s' % (k1,k2)
                if k1 == 'mid' or k1 == 'manga_id':
                    fc = tf.feature_column.crossed_column([k1, k2], hash_bucket_size=20000)
                else:
                    fc = tf.feature_column.crossed_column([k1,k2],hash_bucket_size=200)
                linear_featcols.append(tf.feature_column.indicator_column(fc))
                #dnn_featcols.append(tf.feature_column.embedding_column(fc, dimension=32))




    return featcol, linear_featcols, dnn_featcols


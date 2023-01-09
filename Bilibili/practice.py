from collections import OrderedDict
from datetime import datetime, date, timedelta
import json
import configparser



from model.dnn import dnn
from smart_io.readConf import resolveJson
import os

import tensorflow as tf


def file_lists(pre_fix):
    return pre_fix  + "/*"


def test_lists(pre_fix):
    return pre_fix  + "/*"


def read_table(name, date, table_config,local_path,remote_path):
    print('Reading %s %s'%(name, table_config.get('table_name')))
    # any date offset
    if table_config.get('date_offset') is not None:
      data_date = (datetime(
        int(str(date)[:4]),
        int(str(date)[4:6]),
        int(str(date)[6:8])) + \
          timedelta(days=table_config.get('date_offset'))
        ).strftime('%Y%m%d')
    else:
      data_date = date
    print('Date: %s '%data_date)#w为什么这么设计时间

    fp = os.path.join(
      local_path,
      table_config['table_name'],
      table_config['suffix'])

    if table_config['read_type'] == 'raw':
      data = fp
    elif table_config['read_type'] == 'remote':
      data = os.path.join(
        remote_path,
        table_config['table_name'],
        table_config['suffix'] + '=%s'%data_date
        )
    else:
        raise Exception('read_type: %s not supported.'%(table_config['read_type']))
    return data

if __name__ == '__main__':
    print("sss")

    date0 = (datetime.now() - timedelta(15)).strftime('%Y%m%d')
    config = "conf/conf_dnn.json"
    column_dict = configparser.ConfigParser()
    column_dict.read('conf/columns_2.ini')


    #解析jason文件
    config_map = {}
    with open(config, 'r') as load_f:
        load_dict = json.load(load_f)['conf']
        for k in load_dict:
            config_map[k] = load_dict[k]
    print(config_map)

    column_diction = column_dict

    data_dict = {}
    local_path = config_map['PATH']['local']
    remote_path = config_map['PATH']['remote']

    #read table
    for name, dic in config_map['DATA'].items():
        data = read_table(name,date0,dic,local_path,remote_path)
        data_dict[name] = data
    print(data_dict)

    #feature preprocess1
    column_dict = OrderedDict(column_dict)
    print(column_dict)
    column_dict['FEAT_CROSS'] = {
        k: v.split(',') for k, v in column_dict['FEAT_CROSS'].items()
    }

    print(column_dict)



    # feature preprocess2
    feat_cols = []
    feats = []
    use_col = {}
    cate_cols = []

    feature_columns = OrderedDict()
    features = dict(column_dict['FEAT_TRAIN'])
    for k in  column_dict[ 'FEAT_EXCLUDE']:
        del features[k] #清理掉不需要的feature

    del features['label'] # pop label

    for feat in features:
        # print("feat = ", feat)
        if feat in ['mid']:
            continue
        data_type =  features[feat]

        if data_type ==  'int' or data_type == 'float' or data_type == 'double':
            fc = tf.feature_column.numeric_column(feat)
            feat_cols.append(fc)
            feature_columns[feat] = fc
            use_col[feat] = data_type
        elif data_type == 'string':
            fc = tf.feature_column.categorical_column_with_hash_bucket(key=feat,hash_bucket_size=20000)
            cate_cols.append(feat)
            feat_cols.append(tf.feature_column.indicator_column(fc))
            feature_columns[feat] = fc
            use_col[feat] = data_type


    for k1, k2s in column_dict['FEAT_CROSS'].items():
       for k2 in k2s:
            k = '%s_%s' % (k1,k2)
            fc = tf.feature_column.crossed_column([k1,k2],hash_bucket_size=64)
            feature_columns[k] = fc
            cate_cols.append(k)
            feat_cols.append(tf.feature_column.indicator_column(fc))
    print("-----------------------------")
    print("feature_columns:")
    print(feature_columns["u_norm_pay_amount_app_d"],feature_columns["comic_id"])
    print("use_col")
    print(use_col)
    print("feats:")
    print(feats) #这里是空啊
    print("-----------------------------")

    # 读取训练数据集
    train_files = tf.io.gfile.glob(file_lists(data['FEAT_TRAIN']))
    # 读取测试数据集
    test_files = tf.io.gfile.glob(test_lists(data['FEAT_TEST']))

    estimator = dnn(feature_columns, config_map)
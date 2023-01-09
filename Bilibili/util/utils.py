# coding=utf-8
# /usr/bin/evn python

'''
Author: Yuan Fei
Email: arccos2002@gmail.com
Date: 2019-08-16 15:13
Desc: 
'''

from datetime import datetime, date, timedelta
import time
from collections import OrderedDict
import os

def yesterday():
    return (date.today() + timedelta(days=-2)).strftime("%Y%m%d")


def twodays():
    return (date.today() + timedelta(days=-3)).strftime("%Y%m%d")


def timestamp():
    return str(int(time.time()))


def current_day():
    return (date.today() + timedelta(days=0)).strftime("%Y%m%d")



def read(date,config):
    """
    Read tables in config.
    """
    data_dict = {}
    local_path = config['PATH']['local']
    remote_path = config['PATH']['remote']
    for name, dic in config['DATA'].items():
      data = read_table(name, date, dic,local_path,remote_path)
      data_dict[name] = data
      print('Finished reading section %s.'%name)
    print('Finished reading all sections.')
    return data_dict

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



def column_dict_preprocess(column_dict):
  # split cross column
  column_dict = OrderedDict(column_dict)
  for k, v in column_dict['FEAT_CROSS'].items():
      print("key: ", k)
      print("value:" , v)
  column_dict['FEAT_CROSS'] = {
    k: v.split(',') for k, v in column_dict['FEAT_CROSS'].items()
  }

  print(column_dict['FEAT_CROSS'])
  return column_dict


def genRecord_defaults(column_dict):
    # 生成读取训练集数据的默认类型列表
    uid_col = list(column_dict['FEAT_UID'].keys())[0]

    # print("uid_col=" , uid_col)
    decode_col = []
    decode_col_idx = []
    defaults = []
    for i, (j, tp) in enumerate(column_dict['FEAT_TRAIN'].items()):

        # print("i," ,i)
        # print("j,tp", (j, tp))
        if not j in column_dict['FEAT_EXCLUDE'] or j == uid_col:
            decode_col_idx.append(i)
            decode_col.append(j)
            if tp == 'string':
                defaults.append('0')
            elif tp == 'double':
                defaults.append(0.0)
            elif tp == 'float':
                defaults.append(0.0)
            elif tp == 'int' or tp == 'bigint':
                defaults.append(0)
            elif tp == 'timestamp':
                defaults.append('0')
            else:
                raise Exception('Type %s of No.%s: %s not supported.' % (tp, i, j))
    return decode_col,decode_col_idx,defaults



# coding=utf-8
# /usr/bin/evn python

'''
Author: Yuan Fei
Email: arccos2002@gmail.com
Date: 2019-08-16 14:54
Desc: 
'''

import json
import configparser
from util.utils import *

def   resolveJson(path):
    column_dict = configparser.ConfigParser()
    #column_dict.read('conf/columns_eoe_homepage_label.ini')
    #column_dict.read('conf/columns_eoe_mangaid.ini')
    #column_dict.read('conf/columns_eoe.ini')
    #print(column_dict)

    config_map = {}
    with open(path, 'r') as load_f:

        load_dict = json.load(load_f)['conf']

        for k in load_dict:
            config_map[k] = load_dict[k]
        column_dict.read(config_map['columns'])


    #print(config_map)

    return {
    'TASK_CONFIG': config_map,
    'COLUMN_DICT': column_dict
    }

if __name__ == "__main__":
    pass







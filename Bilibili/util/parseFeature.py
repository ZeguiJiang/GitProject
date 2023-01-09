#coding=utf-8
#/usr/bin/evn python

'''
Author: Fei
Email: arccos2002@gmail.com
Date: 2019-08-13 18:25
Desc: 
'''
import collections

def parser_feature(filename):
    colMap =collections.OrderedDict()

    with open(filename) as f:
        for line in f:
            col, valuetype = line.strip("\n").split("\t")
            colMap[col] = valuetype
    return colMap



if __name__ == "__main__":
    print(
        parser_feature("../conf/feature.conf")
    )

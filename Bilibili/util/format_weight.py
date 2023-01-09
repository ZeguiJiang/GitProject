# coding=utf-8
# /usr/bin/evn python

'''
Author: Fei Xiaoming
Email: arccos2002@gmail.com
Date: 2019-08-16 15:55
Desc:
'''
import numpy as np
import logging
import tensorflow as tf
import os
from util.utils import yesterday
from util.ilog import ilog

#logging.getLogger().setLevel(logging.INFO)
#tf.logging.set_verbosity(tf.logging.INFO)


def format_weight(estimator, flag, known_args, features):

    weight_map = {}
    for name in estimator.get_variable_names():

        print (name,estimator.get_variable_value(name))

        #print("name:",name)
        if 'bias' in name :
           weight_map['bias'] = estimator.get_variable_value(name)[0]
        elif 'kernel' in name:
            for n ,i in enumerate(estimator.get_variable_value(name)):

                weight_map[n] = i[0]


        elif "linear_model" in name and "Ftrl" not in name:
           fe = name.split("/")[-2]
           #print("fe\t\t\t", fe)
           if len(estimator.get_variable_value(name) > 1):
               for a, b in enumerate(estimator.get_variable_value(name)):
                   if isinstance(b, np.float32):
                       # print(name, fe, ":", b)
                       weight_map['bias'] = b
                   else:
                       # print(fe, ":", b[0])
                       weight_map[fe] = b[0]
           else:
               pass
               # print("name: ", fe, "value: ", estimator.get_variable_value(name))
    #print("weight map: \t\t\t\t", weight_map)
    sb1 = (
    ",".join([str(k[0]) + ":" + str(k[1]) for k in sorted(weight_map.items(), key=lambda x: x[1], reverse=True)]))
    sb = (
    "\n".join([str(k[0]) + ":" + str(k[1]) for k in sorted(weight_map.items(), key=lambda x: x[1], reverse=True)]))
    # print(n)
    # n=sorted(n, key=lambda x: x[1],reverse=True)
    # sb="\n".join([str(m[0])+":"+str(m[1])for m in n ])

    #print("sb:" ,sb)
    if flag:
        #path = known_args.d_result_dir
        path = known_args
        if not os.path.isdir(path):
            os.mkdir(path)
        file = path + '/' + yesterday() + ".txt"
        with open(file, "w") as f:
            f.write(sb)
            ilog("write file %s" % file)
    #ilog(sb)
    return sb1

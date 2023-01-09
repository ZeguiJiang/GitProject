from collections import OrderedDict
import numpy as np
import tensorflow as tf

'''
Author: Yuan
Email: arccos2002@gmail.com
Date: 2019-08-16 15:13
Desc: 
'''

def feature_preprocess(raw_features):
  features = {}
  for k in raw_features:
    if 'attention' in k:
      size = int(k[k.find('attention') + 9:])
      s = tf.strings.split(
        raw_features[k], sep='|', result_type='RaggedTensor')\
        .to_tensor()
      for i in range(size):
        features['%s_%s'%(k, i)] = s[:, i]
    if 'list' in k:
      features[k] = tf.strings.split(
        raw_features[k], sep='|', result_type='RaggedTensor')\
        .to_tensor(name=k)
    else:
      features[k] = raw_features[k]
  return features

# coding=utf-8
# /usr/bin/evn python

'''
FileName: export_model.py
Author: pulan
Date: 2020/8/18
Desc:
'''

import os
import shutil
import tensorflow as tf
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2


def export_model(estimator, featcol, configMap):


    feature_dict={}
    warmup_dict={}

    for feat, value_type in featcol.items():
        if value_type in ['float']:
            feature_dict[feat] = tf.placeholder(tf.float32, [1],  name=feat)
            warmup_dict[feat]=tf.make_tensor_proto([0.0],dtype=tf.float32)

        elif value_type in ['int']:
            feature_dict[feat]=tf.placeholder(tf.int32, [1],  name=feat)
            warmup_dict[feat]=tf.make_tensor_proto([0],dtype=tf.int32)

        elif value_type in ['double']:
            feature_dict[feat] = tf.placeholder(tf.float64, [1], name=feat)
            warmup_dict[feat] = tf.make_tensor_proto([0.0], dtype=tf.float64)

        elif value_type in ['bigint', 'timestamp']:
            feature_dict[feat]=tf.placeholder(tf.int64, [1],  name=feat)
            warmup_dict[feat]=tf.make_tensor_proto([0],dtype=tf.int64)

        else:
            feature_dict[feat]=tf.placeholder(tf.string, [1],  name=feat)
            warmup_dict[feat]=tf.make_tensor_proto(['0'],dtype=tf.string)


    print("feature_dict", feature_dict)
    print("warmup_dict", warmup_dict)



    #warmupfile =  "/Users/monarch/Desktop/model_zoo/tf_serving_warmup_requests"
    warmupfile = "/tf_serving_warmup_requests"
    with tf.python_io.TFRecordWriter(warmupfile) as writer:
        request = predict_pb2.PredictRequest(
            model_spec=model_pb2.ModelSpec(name="eoe_rec_dnn", signature_name="serving_default"),

            inputs=warmup_dict
            )
        log = prediction_log_pb2.PredictionLog(
        predict_log=prediction_log_pb2.PredictLog(request=request))
        writer.write(log.SerializeToString())


    example_input_fn = (tf.estimator.export.build_raw_serving_input_receiver_fn(feature_dict))



    estimator.export_savedmodel(export_dir_base=configMap['export_dir'], serving_input_receiver_fn=example_input_fn,
                                assets_extra={'tf_serving_warmup_requests': warmupfile},
                                as_text=True, strip_default_attrs=True)



if __name__ == "__main__":
    pass


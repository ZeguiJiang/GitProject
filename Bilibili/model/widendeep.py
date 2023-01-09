'''
Author: shuwei Huang
Email: x1ao.shu@163.com
Date: 2022-10-21 12:08
Desc: Define Wide N Deep Model
'''

import os
import  tensorflow as tf
import time
from tensorflow.estimator import DNNLinearCombinedClassifier
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def timestamp():
    return str(int(time.time()))

def wdl(linear_featcols, dnn_featcols, configMap):
    model_dir = configMap['model_dir']
    #model_dir = "/summary_dir/"

    run_config = tf.estimator.RunConfig(save_checkpoints_steps=1000)# 每隔多少步检查点
    session_config = tf.ConfigProto(device_count={"CPU": os.cpu_count()},
                              inter_op_parallelism_threads=os.cpu_count(),
                              intra_op_parallelism_threads=os.cpu_count(),
                              log_device_placement=True)
    run_config = run_config.replace(session_config=session_config)
    return DNNLinearCombinedClassifier(
        config = run_config,
        #dnn_dropout=0.5,
        # wide settings
        linear_feature_columns=linear_featcols,
        ##linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.001, l1_regularization_strength=0.001, l2_regularization_strength=0.001),
        linear_optimizer=tf.train.FtrlOptimizer(
            learning_rate=0.1,

            l1_regularization_strength=0.01, 
            l2_regularization_strength=0.001
         ),
        # deep settings
        dnn_feature_columns=dnn_featcols,
        dnn_hidden_units=configMap['dnn_hidden_units'],
        dnn_optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=0.1),
        model_dir=model_dir,
        loss_reduction=tf.losses.Reduction.MEAN,
        )




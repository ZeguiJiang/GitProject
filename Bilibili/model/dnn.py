#coding=utf-8
#/usr/bin/evn python
'''
Author: zegui
Email: zeguijiang@bilibili.com
Date: 2021-11-21 12:08
Desc: Define DNN Model
'''

import tensorflow as tf
import logging
import shutil
import os
from util.utils import timestamp
logging.getLogger().setLevel(logging.INFO)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def dnn_logits(deep_net,params):
    with tf.compat.v1.variable_scope('deep'):
        for units in params['hidden_units']:
            #deep_net = tf.layers.dense(deep_net, units=units, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
            deep_net = tf.layers.dense(deep_net, units=units, activation=tf.nn.relu)
            #deep_net = tf.layers.dropout(deep_net,rate=0.5)
        dnn_logits = tf.layers.dense(deep_net, params['n_classes'], activation=None)
    return dnn_logits

def DNN(features, labels, mode, params):
    with tf.compat.v1.variable_scope('embd'):
        deep_net = tf.compat.v1.feature_column.input_layer(features, params['feature_columns'])

    with tf.compat.v1.variable_scope('wdl_clk'):
        logits = dnn_logits(deep_net,params)

    predictions = tf.nn.softmax(logits,name="predictions")

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            # 'mid': features['mid'],
            # 'comic_id': features['comic_id'],
            'probabilities':predictions
        }
        return tf.estimator.EstimatorSpec(mode,predictions=predictions)
    #define loss
    loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # tensorboard evaluation matric
    accuracy = tf.compat.v1.metrics.accuracy(labels=labels,
                                   predictions=tf.to_float(tf.argmax(predictions,1)),
                                   name='acc_op')
    auc = tf.compat.v1.metrics.auc(labels=labels,
                                   predictions=predictions[:, 1],
                                   name='auc_op')

    metrics = {
            'accuracy': accuracy
            ,'auc': auc}

    tf.compat.v1.summary.scalar('accuracy', accuracy[1])
    tf.compat.v1.summary.scalar('auc', auc[1])
    tf.compat.v1.summary.scalar('loss', loss)

    logging_hook = tf.compat.v1.train.LoggingTensorHook({
                              "step":tf.train.get_global_step(),
                               "loss":loss,
                               "accuracy":accuracy[1],
                               "auc":auc[1]},
                            every_n_iter=2000)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN
    # learning rate decay
    lr =tf.compat.v1.train.exponential_decay(
            learning_rate=0.01,
            global_step=tf.compat.v1.train.get_global_step(),
            decay_steps=5000,
            decay_rate=0.96,staircase=False)

    tf.summary.scalar('lr', lr)
    # optimizer
    dnn_optimizer =  tf.compat.v1.train.ProximalAdagradOptimizer(learning_rate=lr)
    #train step
    train_op = dnn_optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,training_hooks=[logging_hook])



def dnn(feature_columns, configMap):
    # get parameters
    params = {
                 'feature_columns': feature_columns,
                 'hidden_units': configMap['hidden_units'],
                 'n_classes': 2,
                 'learning_rate' : configMap['learning_rate']
             }

    model_dir = configMap['model_dir']

    run_config = tf.estimator.RunConfig(save_checkpoints_steps=4000)
    # session_config = tf.compat.v1.ConfigProto(
    #                                         device_count={'GPU': 1},
    #                                         inter_op_parallelism_threads=0,
    #                                         intra_op_parallelism_threads=0,
    #                                         log_device_placement=False
    #                                     )
    session_config = tf.ConfigProto(device_count={"CPU": os.cpu_count()},
                              inter_op_parallelism_threads=os.cpu_count(),
                              intra_op_parallelism_threads=os.cpu_count(),
                              log_device_placement=False)

    run_config = run_config.replace(session_config=session_config)

    return tf.estimator.Estimator(
        model_fn=DNN,
        params=params,
        model_dir=model_dir,
        config = run_config
        )

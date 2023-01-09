#coding=utf-8
#/usr/bin/evn python
'''
Author: zegui
Email: zeguijiang@bilibili.com
Date: 2022-02-08
Desc: Define DeepFM Model
'''

import tensorflow as tf
import logging
import random
import pandas as pd
import numpy as np
import shutil
import os
from util.utils import timestamp
import shutil
#import sys
import os
import json
import glob
from datetime import date, timedelta







logging.getLogger().setLevel(logging.INFO)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def batch_norm_layer(x, train_phase, batch_norm_decay, scope_bn):

    bn_train = tf.contrib.layers.batch_norm(x, decay= batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=True,  reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay= batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z

def deepfm_logits(deep_net,params):
    pass

def dnn_logits(deep_net,params):
    with tf.compat.v1.variable_scope('deep'):
        for units in params['hidden_units']:
            #deep_net = tf.layers.dense(deep_net, units=units, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
            deep_net = tf.layers.dense(deep_net, units=units, activation=tf.nn.relu)
            #deep_net = tf.layers.dropout(deep_net,rate=0.5)
        dnn_logits = tf.layers.dense(deep_net, params['n_classes'], activation=None)
    return dnn_logits

def DeepFM(features, labels, mode, params):
    """Bulid Model function f(x) for Estimator."""
    # ------hyperparameters----
    field_size = params["field_size"]
    feature_size = params["feature_size"]
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]
    # batch_norm_decay = params["batch_norm_decay"]
    # optimizer = params["optimizer"]
    layers = list(map(int, params["deep_layers"].split(',')))
    dropout = list(map(float, params["dropout"].split(',')))




    optimizer = 'Adagrad'
    batch_norm = False

    # ------bulid weights------
    FM_B = tf.get_variable(name='fm_bias', shape=[1], initializer=tf.constant_initializer(0.0))

    FM_W = tf.get_variable(name='fm_w', shape=[feature_size], initializer=tf.glorot_normal_initializer())
    print("FM_w", FM_W)

    FM_V = tf.get_variable(name='fm_v', shape=[feature_size, embedding_size],
                           initializer=tf.glorot_normal_initializer())

    print("FM_V", FM_V)
    # ------build feaure-------
    feat_ids = []
    for i,j in enumerate(features):
        feat_ids.append(i)
    print(features)

    feat_ids  = tf.constant(feat_ids)
    #feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feature_index')
    feat_wgts = tf.nn.embedding_lookup(FM_W, feat_ids)
    # feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feature_value')
    y_w = tf.reduce_sum(tf.multiply(feat_wgts, features), 1)
    print(feat_ids)
    print(feat_wgts)
    print(y_w)
    #feat_ids = features['feat_ids']
    #feat_ids = tf.reshape(feat_ids, shape=[None, 0])
    feat_vals = features['feat_vals']
    # feat_vals = tf.reshape(feat_vals, shape=[-1, field_size])
    #
    # print("feat_ids",feat_ids)
    #
    # print("feat_vals",feat_vals)
    # feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feature_index')
    # feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feature_value')
    # with tf.variable_scope("First-order"):
    #     feat_wgts = tf.nn.embedding_lookup(FM_W, feat_index)  # None * F * 1
    #     y_w = tf.reduce_sum(tf.multiply(feat_wgts, feat_value), 1)

    print("params['feature_columns']",params['feature_columns']),
    print("features", features),
    with tf.compat.v1.variable_scope('embd'):
        deep_net = tf.compat.v1.feature_column.input_layer(features, params['feature_columns'])

    with tf.compat.v1.variable_scope('deep_net'):
        logits = dnn_logits(deep_net,params)
        y_bias = FM_B * tf.ones_like(logits, dtype=tf.float32)
        logits = logits+y_bias+y_w
    print(logits+y_bias )

    predictions = tf.nn.softmax(logits+y_bias,name="predictions")





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

    # ------build f(x)------
    # with tf.variable_scope("First-order"):
    #     feat_wgts = tf.nn.embedding_lookup(FM_W, feat_ids)  # None * F * 1
    #     y_w = tf.reduce_sum(tf.multiply(feat_wgts, feat_vals), 1)
    # #
    # # print("y_w",y_w)
    #
    # with tf.variable_scope("Second-order"):
    #     embeddings = tf.nn.embedding_lookup(FM_V, feat_ids)  # None * F * K
    #     feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])
    #     embeddings = tf.multiply(embeddings, feat_vals)  # vij*xi
    #     sum_square = tf.square(tf.reduce_sum(embeddings, 1))
    #     square_sum = tf.reduce_sum(tf.square(embeddings), 1)
    #     y_v = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1)  # None * 1
    #
    # print(y_v,"y_v")

    # with tf.variable_scope("Deep-part"):
    #     if batch_norm:
    #         # normalizer_fn = tf.contrib.layers.batch_norm
    #         # normalizer_fn = tf.layers.batch_normalization
    #         if mode == tf.estimator.ModeKeys.TRAIN:
    #             train_phase = True
    #             # normalizer_params = {'decay': batch_norm_decay, 'center': True, 'scale': True, 'updates_collections': None, 'is_training': True, 'reuse': None}
    #         else:
    #             train_phase = False
    #             # normalizer_params = {'decay': batch_norm_decay, 'center': True, 'scale': True, 'updates_collections': None, 'is_training': False, 'reuse': True}
    #     else:
    #         normalizer_fn = None
    #         normalizer_params = None
    #
    #     deep_inputs = tf.reshape(embeddings, shape=[-1, field_size * embedding_size])  # None * (F*K)
    #     for i in range(len(layers)):
    #         # if FLAGS.batch_norm:
    #         #    deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn='bn_%d' %i)
    #         # normalizer_params.update({'scope': 'bn_%d' %i})
    #         deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[i], weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='mlp%d' % i)
    #         if batch_norm:
    #             deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, batch_norm_decay= 0.9,scope_bn='bn_%d' % i)
    #         if mode == tf.estimator.ModeKeys.TRAIN:
    #             deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[i])
    #             # Apply Dropout after all BN layers and set dropout=0.8(drop_ratio=0.2)
    #             # deep_inputs = tf.layers.dropout(inputs=deep_inputs, rate=dropout[i], training=mode == tf.estimator.ModeKeys.TRAIN)
    #
    #     y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity, \
    #                                                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
    #                                                scope='deep_out')
    #     y_d = tf.reshape(y_deep, shape=[-1])


    # with tf.variable_scope("DeepFM-out"):
    #     y_bias = FM_B * tf.ones_like(y_d, dtype=tf.float32)  # None * 1
    #     y = y_bias + y_w + y_v + y_d
    #     pred = tf.sigmoid(y)

    # predictions = {"prob": predictions}
    # export_outputs = {
    #     tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
    #         predictions)}
    # # Provide an estimator spec for `ModeKeys.PREDICT`
    # if mode == tf.estimator.ModeKeys.PREDICT:
    #     return tf.estimator.EstimatorSpec(
    #         mode=mode,
    #         predictions=predictions,
    #         export_outputs=export_outputs)
    #
    # # ------bulid loss------
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)) + \
    #        l2_reg * tf.nn.l2_loss(FM_W) + \
    #        l2_reg * tf.nn.l2_loss(FM_V)  # + \ l2_reg * tf.nn.l2_loss(sig_wgts)
    #
    # # Provide an estimator spec for `ModeKeys.EVAL`
    # eval_metric_ops = {
    #     "auc": tf.metrics.auc(labels, predictions)
    # }
    # if mode == tf.estimator.ModeKeys.EVAL:
    #     return tf.estimator.EstimatorSpec(
    #         mode=mode,
    #         predictions=predictions,
    #         loss=loss,
    #         eval_metric_ops=eval_metric_ops)
    #
    # # ------bulid optimizer------
    # if optimizer == 'Adam':
    #     optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    # elif optimizer == 'Adagrad':
    #     optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    # elif optimizer == 'Momentum':
    #     optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    # elif optimizer == 'ftrl':
    #     optimizer = tf.train.FtrlOptimizer(learning_rate)
    #
    # train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    #
    # # Provide an estimator spec for `ModeKeys.TRAIN` modes
    # if mode == tf.estimator.ModeKeys.TRAIN:
    #     return tf.estimator.EstimatorSpec(
    #         mode=mode,
    #         predictions=predictions,
    #         loss=loss,
    #         train_op=train_op)


def deepfm(feature_columns, configMap):
    # get parameters

    params = {
                 'feature_columns': feature_columns,
                 'n_classes': 2,
                 'learning_rate' : configMap['learning_rate'],
                 "feature_size": configMap['feature_size'],
                 "embedding_size": configMap['embedding_size'],
                 "l2_reg": configMap['l2_reg'],
                 "deep_layers": "256,128,64",
                 "dropout": "0.6,0.5,0.5",
                 "batch_norm_decay" : configMap['batch_norm_decay'],
                "hidden_units": configMap['hidden_units'],
                 "field_size" : 0

             }




    model_dir = configMap['model_dir']

    run_config = tf.estimator.RunConfig(save_checkpoints_steps=4000)
    # session_config = tf.compat.v1.ConfigProto(
    #                                         device_count={'GPU': 1},
    #                                         inter_op_parallelism_threads=0,
    #                                         intra_op_parallelism_threads=0,
    #                                         log_device_placement=False
    #                                     )

    # config = tf.estimator.RunConfig().replace(
    #     session_config=tf.ConfigProto(device_count={'GPU': 0, 'CPU': FLAGS.num_threads}),
    #     log_step_count_steps=FLAGS.log_steps, save_summary_steps=FLAGS.log_steps)

    session_config = tf.ConfigProto(device_count={"CPU": os.cpu_count()},
                              inter_op_parallelism_threads=os.cpu_count(),
                              intra_op_parallelism_threads=os.cpu_count(),
                              log_device_placement=False)

    run_config = run_config.replace(session_config=session_config)

    return tf.estimator.Estimator(
        model_fn = DeepFM,
        params=params,
        model_dir=model_dir,
        config = run_config
        )

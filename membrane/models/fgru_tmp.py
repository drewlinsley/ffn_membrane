#!/usr/bin/env python
import os
import argparse
import numpy as np
import tensorflow as tf
from membrane.membrane_ops import mops as model_fun
from membrane.layers.feedforward import conv
from membrane.layers.feedforward import normalization
# from layers.feedforward import pooling
from membrane.layers.recurrent import feedback_hgru


def experiment_params(
        train_name=None,
        test_name=None,
        train_shape=None,
        test_shape=None,
        affinity=12,
        gt_idx=1,
        z=18):
    """Parameters for the experiment."""
    if train_shape is None:
        train_shape = [0]
    exp = {
        'lr': [1e-2],
        'loss_function': ['cce'],
        'optimizer': ['nadam'],
        'training_routine': ['seung'],
        'train_dataset': [train_name],
        'test_dataset': [test_name],
        'affinity': [affinity],
        'cross_val': {'train': [0, 80], 'test': [80, 100]},
        'gt_idx': [gt_idx],  # Changes based on affinity
        'train_input_shape': [z] + train_shape + [1],
        'train_label_shape': [z] + train_shape + [affinity],
        'test_input_shape': [z] + test_shape + [1],
        'test_label_shape': [z] + test_shape + [affinity],
        'train_stride': [1, 1, 1],
        'test_stride': [1, 1, 1],
        'tf_dtype': tf.float32,
        'np_dtype': np.float32
    }
    exp['exp_label'] = __file__.split('.')[0].split(os.path.sep)[-1]
    exp['train_augmentations'] = [
        {'min_max_native_normalization': []},
        # {'normalize_volume': lambda x: x / 255.},
        # {'warp': {}},
        {'random_crop': []},
        {'pixel': {}},
        {'misalign': {}},
        {'blur': {}},
        {'missing': {}},
        {'flip_lr': []},
        {'flip_ud': []},
    ]
    exp['test_augmentations'] = [
        # {'normalize_volume': lambda x: x / 255.}
        {'min_max_native_normalization': []},
        {'random_crop': []},
    ]
    exp['train_batch_size'] = 1  # Train/val batch size.
    exp['test_batch_size'] = 1  # Train/val batch size.
    exp['top_test'] = 5  # Keep this many checkpoints/predictions
    exp['epochs'] = 100000
    exp['save_weights'] = False  # True
    exp['test_iters'] = 1000
    exp['shuffle_test'] = False  # Shuffle test data.
    exp['shuffle_train'] = True
    return exp


def build_model(data_tensor, reuse, training, output_channels):
    """Create the hgru from Learning long-range..."""
    filters = [14]
    kernel_size = [1, 5, 5]
    with tf.variable_scope('cnn', reuse=reuse):
        # Unclear if we should include l0 in the down/upsample cascade
        with tf.variable_scope('in_embedding', reuse=reuse):
            in_emb = conv.conv3d_layer(
                bottom=data_tensor,
                name='l0',
                stride=[1, 1, 1],
                padding='SAME',
                num_filters=filters[0],
                kernel_size=[1, 5, 5],
                trainable=training,
                use_bias=True)
            in_emb = tf.nn.elu(in_emb)

        # hGRU down
        layer_hgru = feedback_hgru.hGRU(
            layer_name='hgru_1',
            x_shape=in_emb.get_shape().as_list(),
            timesteps=8,
            h_ext=[[1, 7, 7], [3, 7, 7], [1, 1, 1]],
            strides=[1, 1, 1, 1, 1],
            pool_strides=[1, 4, 4],
            padding='SAME',
            aux={
                'symmetric_weights': True,
                'dilations': [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                'batch_norm': True,
                'pooling_kernel': [1, 4, 4],
                'intermediate_ff': filters,  # + filters,
                'intermediate_ks': [kernel_size]},
            train=training)
        h2 = layer_hgru.build(in_emb)
        nh2 = normalization.batch(
            bottom=h2,
            name='hgru_bn',
            fused=True,
            renorm=True,
            training=training)
        with tf.variable_scope('out_embedding', reuse=reuse):
            out_emb = conv.conv3d_layer(
                bottom=nh2,
                name='out_emb',
                stride=[1, 1, 1],
                padding='SAME',
                num_filters=output_channels,
                kernel_size=kernel_size,
                trainable=training,
                use_bias=True)
    return out_emb


def main(
        train=None,
        test=None,
        row_id=None,
        gpu_device='/gpu:0',
        evaluate=False,
        checkpoint=None,
        full_volume=False,
        force_meta=None,
        full_eval=False,
        bethge=None,
        adabn=False,
        test_input_shape=False,
        test_label_shape=False,
        overwrite_training_params=False,
        z=18):
    """Run an experiment with hGRUs."""
    version = '3d'
    tf_records = False
    if evaluate:
        return model_fun.evaluate_model(
            test=test,
            gpu_device=gpu_device,
            z=z,
            version=version,
            build_model=build_model,
            experiment_params=experiment_params,
            checkpoint=checkpoint,
            force_meta=force_meta,
            full_volume=full_volume,
            full_eval=full_eval,
            test_input_shape=test_input_shape,
            test_label_shape=test_label_shape,
            adabn=adabn,
            bethge=bethge,
            tf_records=tf_records)
    else:
        raise NotImplementedError

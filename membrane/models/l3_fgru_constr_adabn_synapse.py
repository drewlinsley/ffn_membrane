#!/usr/bin/env python
import os
import argparse
import numpy as np
import tensorflow as tf
from membrane.membrane_ops import mops as model_fun
from membrane.layers.feedforward import conv
from membrane.layers.feedforward import normalization
from membrane.layers.recurrent import feedback_hgru_constrained_3l as feedback_hgru
from membrane.membrane_ops import tf_fun



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
        'lr': [1e-3],
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
        {'pixel': {}},
        {'misalign': {}},
        {'blur': {}},
        {'missing': {}},
        {'flip_lr': []},
        {'flip_ud': []},
        {'random_crop': []},
    ]
    exp['test_augmentations'] = [
        # {'normalize_volume': lambda x: x / 255.}
        {'min_max_native_normalization': []},
        {'random_crop': []},
    ]
    exp['train_batch_size'] = 1  # Train/val batch size.
    exp['test_batch_size'] = 1  # Train/val batch size.
    exp['top_test'] = 1  # Keep this many checkpoints/predictions
    exp['epochs'] = 100000
    exp['save_weights'] = False  # True
    exp['test_iters'] = 1000
    exp['shuffle_test'] = False  # Shuffle test data.
    exp['shuffle_train'] = True
    exp['adabn'] = True
    return exp


def synapse_experiment_params(
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
    reader_image_shape = [160, 160, 160, 2]
    reader_label_shape = [160, 160, 160, 2]
    exp = {
        'lr': [1e-3],
        'loss_function': ['cce'],
        'optimizer': ['nadam'],
        'training_routine': ['seung'],
        'train_input_shape': [96, 96, 96, 2],
        'train_label_shape': [96, 96, 96, 1],  # JUST RIBBONS
        'test_input_shape': [128, 128, 128, 2],
        'test_label_shape': [128, 128, 128, 1],
        'train_stride': [1, 1, 1],
        'test_stride': [1, 1, 1],
        'tf_dtype': tf.float32,
        'np_dtype': np.float32
    }
    exp['exp_label'] = __file__.split('.')[0].split(os.path.sep)[-1]
    exp['tf_reader'] = {
        'volume': {
            'dtype': tf.float32,
            'reshape': reader_image_shape  # exp['train_input_shape']
        },
        'label': {
            'dtype': tf.float32,
            'reshape': reader_label_shape  # exp['train_label_shape']
        }
    }
    exp['tf_dict'] = {
        'volume': tf_fun.fixed_len_feature(dtype='string'),
        'label': tf_fun.fixed_len_feature(dtype='string')
    }
    exp['train_augmentations'] = [
        # {'min_max_native_normalization': []},
        # {'normalize_volume': lambda x: x / 255.},
        # {'warp': {}},
        {'smooth': []},
        {'random_crop': []},
        {'flip_lr': []},
        {'flip_ud': []},
        {'flip_depth': []},
    ]
    exp['test_augmentations'] = [
        # {'min_max_native_normalization': []},
        # {'smooth': []},
        {'center_crop': []},
        # {'normalize_volume': lambda x: x / 255.}
    ]
    exp['train_batch_size'] = 4   # Train/val batch size.
    exp['test_batch_size'] = 1  # Train/val batch size.
    exp['top_test'] = 1  # Keep this many checkpoints/predictions
    exp['epochs'] = 100000  # 5
    exp['save_weights'] = False  # True
    exp['test_iters'] = 1000  # 1
    exp['shuffle_test'] = False  # Shuffle test data.
    exp['shuffle_train'] = True
    exp['adabn'] = True
    return exp


def build_model(data_tensor, reuse, training, output_channels, merge="none"):
    """Create the hgru from Learning long-range..."""
    filters = [28]
    kernel_size = [1, 5, 5]
    up_kernel = [1, 2, 2]

    im_data_tensor = tf.expand_dims(data_tensor[..., 0], axis=-1)  # ONLY IMAGE
    im_data_tensor = im_data_tensor / 255.
    with tf.variable_scope('cnn', reuse=reuse):
        # Unclear if we should include l0 in the down/upsample cascade
        with tf.variable_scope('in_embedding', reuse=reuse):
            in_emb = conv.conv3d_layer(
                bottom=im_data_tensor,
                name='l0',
                stride=[1, 1, 1],
                padding='SAME',
                num_filters=filters[0],
                kernel_size=[1, 5, 5],
                trainable=training,
                use_bias=True)
            in_emb_pre = tf.nn.elu(in_emb)

        # Downsample
        in_emb = tf.concat([in_emb_pre, tf.expand_dims(data_tensor[..., 1], axis=-1)], axis=-1)
        in_emb = conv.down_block_v2(
            layer_name='l1',
            bottom=in_emb,
            kernel_size=kernel_size,
            num_filters=filters[0],
            training=training,
            norm_type="batch_norm",
            reuse=reuse)

        # hGRU down
        layer_hgru = feedback_hgru.hGRU(
            layer_name='hgru_1',
            x_shape=in_emb.get_shape().as_list(),
            timesteps=2,
            h_ext=[[1, 9, 9], [3, 7, 7], [3, 5, 5], [1, 1, 1], [1, 1, 1]],
            strides=[1, 1, 1, 1, 1],
            pool_strides=[1, 2, 2],
            padding='SAME',
            aux={
                'symmetric_weights': True,
                'dilations': [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                'batch_norm': True,
                'pooling_kernel': [1, 2, 2],
                'intermediate_ff': filters,  # + filters,
                'intermediate_ks': [kernel_size]},
            train=training)
        nh2 = layer_hgru.build(in_emb)

    with tf.variable_scope('cnn_readout', reuse=reuse):
        nh2 = normalization.batch(
            bottom=nh2,
            name='hgru_bn',
            fused=True,
            renorm=True,
            training=training)
        ul0 = conv.up_block_v2(
            layer_name='ul0',
            bottom=nh2,
            skip_activity=in_emb_pre,
            kernel_size=up_kernel,
            num_filters=filters[0],  # MEMBRANE
            training=training,
            norm_type=None,  # norm_type,
            reuse=reuse)
        with tf.variable_scope('out_embedding', reuse=reuse):
            out_emb = conv.conv3d_layer(
                bottom=ul0,
                name='out_emb',
                stride=[1, 1, 1],
                padding='SAME',
                num_filters=output_channels,
                kernel_size=[1, 1, 1],
                trainable=training,
                use_bias=True)
    # out_emb = tf.sigmoid(out_emb)
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
        return_sess=False,
        return_restore_saver=False,
        train_input_shape=None,
        train_label_shape=None,
        test_input_shape=None,
        test_label_shape=None,
        summary_dir=None,
        overwrite_training_params=False,
        tf_records=False,
        z=18):
    """Run an experiment with hGRUs."""
    version = '3d'
    if evaluate:
        return model_fun.evaluate_model(
            test=test,
            gpu_device=gpu_device,
            z=z,
            # force_return_model=force_return_model,
            version=version,
            build_model=build_model,
            experiment_params=experiment_params,
            checkpoint=checkpoint,
            force_meta=force_meta,
            full_volume=full_volume,
            full_eval=full_eval,
            return_sess=return_sess,
            test_input_shape=test_input_shape,
            test_label_shape=test_label_shape,
            bethge=bethge,
            tf_records=tf_records)
    else:
        return model_fun.train_model(
            train=train,
            test=test,
            row_id=row_id,
            gpu_device=gpu_device,
            z=z,
            summary_dir=summary_dir,
            version=version,
            checkpoint=checkpoint,
            train_input_shape=train_input_shape,
            train_label_shape=train_label_shape,
            test_input_shape=test_input_shape,
            test_label_shape=test_label_shape,
            build_model=build_model,
            return_restore_saver=return_restore_saver,
            adabn=adabn,
            experiment_params=synapse_experiment_params,
            overwrite_training_params=overwrite_training_params,
            tf_records=tf_records)


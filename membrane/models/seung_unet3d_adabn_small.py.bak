#!/usr/bin/env python
import os
import argparse
import numpy as np
import tensorflow as tf
from membrane.membrane_ops import mops as model_fun
from membrane.layers.feedforward import conv
from membrane.layers.feedforward import normalization
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
        # 'test_input_shape': [z] + test_shape + [1],
        # 'test_label_shape': [z] + test_shape + [affinity],
        'train_stride': [1, 1, 1],
        'test_stride': [1, 1, 1],
        'tf_dtype': tf.float32,
        'np_dtype': np.float32
    }
    exp['exp_label'] = __file__.split('.')[0].split(os.path.sep)[-1]
    exp['train_augmentations'] = [
        # {'min_max_native_normalization': []},
        # {'normalize_volume': lambda x: x / 255.},
        # {'warp': {}},
        {'random_crop': []},
        {'smooth': []},
        {'flip_lr': []},
        {'flip_ud': []},
        {'flip_depth': []},
    ]
    exp['test_augmentations'] = [
        # {'min_max_native_normalization': []},
        {'center_crop': []},
        # {'smooth': []},
        # {'normalize_volume': lambda x: x / 255.}
    ]
    exp['train_batch_size'] = 1  # Train/val batch size.
    exp['test_batch_size'] = 1  # Train/val batch size.
    exp['top_test'] = 1  # Keep this many checkpoints/predictions
    exp['epochs'] = 100000  # 5
    exp['save_weights'] = False  # True
    exp['test_iters'] = 1000  # 1
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
        # {'smooth': []},
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
    exp['train_batch_size'] = 2  # Train/val batch size.
    exp['test_batch_size'] = 1  # Train/val batch size.
    exp['top_test'] = 1  # Keep this many checkpoints/predictions
    exp['epochs'] = 100000  # 5
    exp['save_weights'] = False  # True
    exp['test_iters'] = 1000  # 1
    exp['shuffle_test'] = False  # Shuffle test data.
    exp['shuffle_train'] = True
    exp['adabn'] = True
    return exp


def build_model(
        data_tensor,
        reuse,
        training,
        output_channels,
        norm_type='group'):
    """Create the hgru from Learning long-range..."""
    conv_kernel = [
        [1, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
    ]
    print("Output channels: {}".format(output_channels))
    up_kernel = [1, 2, 2]
    filters = [28, 36, 48, 64, 80]
    # filters = [14, 18, 24, 32, 80]
    # filters = np.asarray(filters) * 2
    data_shape = data_tensor.get_shape().as_list()
    if data_shape[0] is None:
        print("Forcing None in first dim to = 1.")
        data_shape[0] = 1
        data_tensor.set_shape(data_shape)
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
            in_emb = tf.nn.elu(in_emb)

        # Downsample
        in_emb = tf.concat([in_emb, tf.expand_dims(data_tensor[..., 1], axis=-1)], axis=-1)
        l1 = conv.down_block_v2(
            layer_name='l1',
            bottom=in_emb,
            kernel_size=conv_kernel,
            num_filters=filters[1],
            training=training,
            norm_type=norm_type,
            reuse=reuse)
        l2 = conv.down_block_v2(
            layer_name='l2',
            bottom=l1,
            kernel_size=conv_kernel,
            num_filters=filters[2],
            training=training,
            norm_type=norm_type,
            reuse=reuse)
        l3 = conv.down_block_v2(
            layer_name='l3',
            bottom=l2,
            kernel_size=conv_kernel,
            num_filters=filters[3],
            training=training,
            norm_type=norm_type,
            reuse=reuse)
        l4 = conv.down_block_v2(
            layer_name='l4',
            bottom=l3,
            kernel_size=conv_kernel,
            num_filters=filters[4],
            training=training,
            norm_type=norm_type,
            reuse=reuse)

        # Upsample
        ul3 = conv.up_block_v2(
            layer_name='ul3',
            bottom=l4,
            skip_activity=l3,
            kernel_size=up_kernel,
            num_filters=filters[3],
            training=training,
            norm_type=norm_type,
            reuse=reuse)
        ul3 = conv.down_block_v2(
            layer_name='ul3_d',
            bottom=ul3,
            kernel_size=conv_kernel,
            num_filters=filters[3],
            training=training,
            norm_type=norm_type,
            reuse=reuse,
            include_pool=False)
        ul2 = conv.up_block_v2(
            layer_name='ul2',
            bottom=ul3,
            skip_activity=l2,
            kernel_size=up_kernel,
            num_filters=filters[2],
            training=training,
            norm_type=norm_type,
            reuse=reuse)
        ul2 = conv.down_block_v2(
            layer_name='ul2_d',
            bottom=ul2,
            kernel_size=conv_kernel,
            num_filters=filters[2],
            training=training,
            norm_type=norm_type,
            reuse=reuse,
            include_pool=False)
        ul1 = conv.up_block_v2(
            layer_name='ul1',
            bottom=ul2,
            skip_activity=l1,
            kernel_size=up_kernel,
            num_filters=filters[1],
            training=training,
            norm_type=norm_type,
            reuse=reuse)
        ul1 = conv.down_block_v2(
            layer_name='ul1_d',
            bottom=ul1,
            kernel_size=conv_kernel,
            num_filters=filters[1],
            training=training,
            norm_type=norm_type,
            reuse=reuse,
            include_pool=False)
        ul0 = conv.up_block_v2(
            layer_name='ul0',
            bottom=ul1,
            skip_activity=in_emb,
            kernel_size=up_kernel,
            num_filters=filters[0] + 1,  # MEMBRANE
            training=training,
            norm_type=None,  # norm_type,
            reuse=reuse)

        with tf.variable_scope('out_embedding', reuse=reuse):
            """
            out_emb = conv.conv3d_layer(
                bottom=ul0,
                name='out_emb',
                stride=[1, 1, 1],
                padding='SAME',
                num_filters=output_channels,
                kernel_size=[1, 5, 5],
                trainable=training,
                use_bias=True)
            """
            out_emb = tf.layers.conv3d(
                inputs=ul0,
                filters=output_channels,
                kernel_size=[1, 5, 5],
                strides=[1, 1, 1],
                padding="SAME",
                use_bias=True)
    return out_emb
    # return tf.sigmoid(out_emb)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train',
        dest='train',
        type=str,
        default=None,
        help='Train data.')
    parser.add_argument(
        '--test',
        dest='test',
        type=str,
        default=None,
        help='Test data.')
    args = parser.parse_args()
    main(**vars(args))

import os
import re
import sys
import json
import numpy as np
import tensorflow as tf
from scipy import signal


# Hack to put BSDS on the path
sys.path.append(
    os.path.join(
        '/media',
        'data_cifs',
        'cluster_projects',
        'BSDS500',
        'py-bsds500'))


def update_config(param_dict, config):
    """Update config with finetuning params."""
    for k, v in param_dict.iteritems():
        setattr(config, k, v)
    return config


def strip_coors(data):
    """Strip coordinates from briggman data."""
    coords = []
    for d in data:
        x = d.split(os.path.sep)
        split_x = re.split('(?<=x)0+', x[-4])
        split_y = re.split('(?<=y)0+', x[-3])
        split_z = re.split('(?<=z)0+', x[-2])
        if len(split_x[1]):
            split_x = int(split_x[1])
        else:
            split_x = 0
        if len(split_y[1]):
            split_y = int(split_y[1])
        else:
            split_y = 0
        if len(split_z[1]):
            split_z = int(split_z[1])
        else:
            split_z = 0
        coords += [[split_x, split_y, split_z]]
    return np.asarray(coords)


def update_lr(it_lr, alg, test_losses, lr_info=None):
    """Update learning rate according to an algorithm."""
    if lr_info is None:
        lr_info = {}
    if alg == 'seung':
        threshold = 10
        if 'change' not in lr_info.keys():
            lr_info['change'] = 0
        if lr_info['change'] >= 4:
            return it_lr, lr_info
        # Smooth test_losses then check to see if they are still decreasing
        if len(test_losses) > threshold:
            smooth_test = signal.savgol_filter(np.asarray(test_losses), 3, 2)
            check_test = np.all(np.diff(smooth_test)[-threshold:] < 0)
            if check_test:
                it_lr = it_lr / 2.
            lr_info['change'] += 1
        return it_lr, lr_info
    elif alg is None or alg == '' or alg == 'none':
        return it_lr, lr_info
    else:
        raise NotImplementedError('No routine for: %s' % alg)


def count_parameters(var_list, print_count=False):
    """Count the parameters in a tf model."""
    params = []
    for v in var_list:
        if 'horizontal' in v.name:
            count = np.maximum(np.prod(
                [x for x in v.get_shape().as_list()
                    if x > 1]), 1)
            count = (count / 2) + v.get_shape().as_list()[-1]
            params += [count]
        else:
            params += [
                np.maximum(
                    np.prod(
                        [x for x in v.get_shape().as_list()
                            if x > 1]), 1)]
    param_list = [
        (p, v.get_shape().as_list()) for p, v in zip(
            params, var_list)]
    if print_count:
        print json.dumps(param_list, indent=4)
    return np.sum(params)


def check_shapes(scores, labels):
    """Check and fix the shapes of scores and labels."""
    if not isinstance(scores, list):
        if len(
                scores.get_shape()) != len(
                    labels.get_shape()):
            score_shape = scores.get_shape().as_list()
            label_shape = labels.get_shape().as_list()
            if len(
                score_shape) == 2 and len(
                    label_shape) == 1 and score_shape[-1] == 1:
                labels = tf.expand_dims(labels, axis=-1)
            elif len(
                score_shape) == 2 and len(
                    label_shape) == 1 and score_shape[-1] == 1:
                scores = tf.expand_dims(scores, axis=-1)
    return scores, labels


def bytes_feature(values):
    """Bytes features for writing TFRecords."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    """Int64 features for writing TFRecords."""
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_feature(values):
    """Float features for writing TFRecords."""
    if isinstance(values, np.ndarray):
        values = [v for v in values]
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def fixed_len_feature(length=[], dtype='int64'):
    """Features for reading TFRecords."""
    if dtype == 'int64':
        return tf.FixedLenFeature(length, tf.int64)
    elif dtype == 'int32':
        return tf.FixedLenFeature(length, tf.int32)
    elif dtype == 'string':
        return tf.FixedLenFeature(length, tf.string)
    elif dtype == 'float':
        return tf.FixedLenFeature(length, tf.float32)
    else:
        raise RuntimeError('Cannot understand the fixed_len_feature dtype.')


def image_summaries(
        images,
        tag):
    """Wrapper for creating tensorboard image summaries.

    Parameters
    ----------
    images : tensor
    tag : str
    """
    im_shape = [int(x) for x in images.get_shape()]
    tag = '%s images' % tag
    if im_shape[-1] <= 3 and (
            len(im_shape) == 3 or len(im_shape) == 4):
        tf.summary.image(tag, images)
    elif im_shape[-1] <= 3 and len(im_shape) == 5:
        # Spatiotemporal image set
        res_ims = tf.reshape(
            images,
            [im_shape[0] * im_shape[1]] + im_shape[2:])
        tf.summary.image(tag, res_ims)


def check_early_stop(
        perf_history,
        minimum_length=20,
        short_history=3,
        long_history=5,
        fail_function=np.less_equal):
    """
    Determine whether to stop early. Using deepgaze criteria:

    We determine this point by comparing the performance from
    the last three epochs to the performance five epochs before those.
    Training runs for at least 20 epochs, and is terminated if all three
    of the last epochs show decreased performance or if
    800 epochs are reached.

    """
    if len(perf_history) < minimum_length:
        early_stop = False
    else:
        short_perf = perf_history[-short_history:]
        long_perf = perf_history[-long_history + short_history:short_history]
        short_check = fail_function(np.mean(long_perf), short_perf)
        if all(short_check):  # If we should stop
            early_stop = True
        else:
            early_stop = False

    return early_stop


def finetune_splits(dataset, reduction, default_cv=0.8):
    """Reduction is 1, 2, 4, or 8 (denominator)."""
    if dataset == 'berson3d_0':
        default_z = int(384 * default_cv)
        if reduction == 1:
            return (default_z, 192, 384)
        elif reduction == 2:
            return (192, 192, 384)
        elif reduction == 4:
            return (192, 192, 192)
        elif reduction == 8:
            return ((96, 192), 192, 192)
        elif reduction == 16:
            return ((96, 192), (96, 192), 192)
        elif reduction == 32:
            return ((96, 192), (96, 192), (96, 192))
        elif reduction == 64:
            return ((144, 192), (96, 192), (96, 192))
        elif reduction == -1:
            return ((96, 114), (112, 272), (112, 272))
        elif reduction == -2:
            return ((96, 132), (112, 272), (112, 272))
        elif reduction == -3:
            return ((96, 150), (112, 272), (112, 272))
        else:
            raise NotImplementedError('Reduction: %s' % reduction)
    elif dataset == 'isbi_20133d_0':
        default_z = int(100 * default_cv)
        if reduction == 1:
            return (default_z, 512, 1024)
        elif reduction == 2:
            return (default_z, 512, 512)
        elif reduction == 4:
            return (default_z, (256, 512), 512)
        elif reduction == 8:
            return (default_z, (256, 512), (256, 512))
        elif reduction == 16:
            return (50, (256, 512), (256, 512))
        elif reduction == 32:
            return (50, (384, 512), (256, 512))
        elif reduction == 64:
            return (50, (384, 512), (384, 512))
        else:
            raise NotImplementedError('Reduction: %s' % reduction)
    elif 'cremi' in dataset:
        default_z = int(125 * default_cv)
        if reduction == 1:
            return (default_z, 625, 1248)
        elif reduction == 2:
            return (default_z, 625, 625)
        elif reduction == 4:
            return (default_z, (312, 625), 625)
        elif reduction == 8:
            return (default_z, (312, 625), (312, 625))
        elif reduction == 16:
            return (62, (312, 625), (312, 625))
        elif reduction == 32:
            return (62, (468, 625), (312, 625))
        elif reduction == 64:
            return (62, (468, 625), (468, 625))
        elif reduction == -1:
            return ((18, 36), (465, 625), (465, 625))
        elif reduction == -2:
            return ((18, 54), (465, 625), (465, 625))
        elif reduction == -3:
            return ((18, 72), (465, 625), (465, 625))
        else:
            raise NotImplementedError('Reduction: %s' % reduction)
    else:
        raise NotImplementedError


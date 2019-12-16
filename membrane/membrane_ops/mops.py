
import os
import numpy as np
import tensorflow as tf
# from config import Config
from membrane.utils import py_utils
from membrane.membrane_ops import mtraining as training
# from ops import metrics
# from membrane.membrane_ops import data_structure
# from membrane.membrane_ops import data_loader_queues as data_loader
from membrane.membrane_ops import optimizers
from membrane.membrane_ops import tf_fun
from membrane.membrane_ops import gradients


WEIGHT_DECAY = 1e-4


def f1_metric(y_true, y_pred, eps=1e-8):
    true_positives = tf.sum(
        tf.round(tf.minimum(tf.maximum(y_true * y_pred, 0), 1)))
    possible_positives = tf.sum(
        tf.round(tf.minimum(tf.maximum(y_true, 0), 1)))
    predicted_positives = tf.sum(
        tf.round(tf.minimum(tf.maximum(y_pred, 0), 1)))
    precision = true_positives / (predicted_positives + eps)
    recall = true_positives / (possible_positives + eps)
    f1_val = 2 * (precision * recall) / (precision + recall + eps)
    return f1_val, precision, recall


def configure_model(
        train=None,
        test=None,
        row_id=None,
        gpu_device='/gpu:0',
        z=18,
        version='3d',
        build_model=None,
        experiment_params=None,
        force_meta=None,
        evaluate=False):
    """Run configuration routines."""
    config = Config()
    assert build_model is not None, 'Pass a model function.'
    assert experiment_params is not None, 'Pass experiment_params.'

    # Prepare train data
    if not evaluate:
        if train is not None:
            config.train_dataset = train
        else:
            if version == '3d':
                config.train_dataset = 'berson3d'
            else:
                config.train_dataset = 'berson'
        train_dataset_module = py_utils.import_module(
            model_dir=config.dataset_info,
            dataset=config.train_dataset)
        train_dataset_module = train_dataset_module.data_processing(
            z_slices=z)
        meta_path = os.path.join(
            config.tf_records,
            'class_weights_%s.npz' % train_dataset_module.output_name)
        if os.path.exists(meta_path):
            train_data_meta = np.load(meta_path)
        else:
            train_data_meta = None
    else:
        train_dataset_module, train_data_meta = None, None

    # Prepare test data
    if test is not None:
        config.test_dataset = test
    else:
        if version == '3d':
            config.test_dataset = 'berson3d'
        else:
            config.test_dataset = 'berson'
    try:
        test_dataset_module = py_utils.import_module(
            model_dir=config.dataset_info,
            dataset=config.test_dataset)
        test_dataset_module = test_dataset_module.data_processing(
            z_slices=z)
    except Exception:
        print 'Falling back to Berson defaults'
        test_dataset_module = py_utils.import_module(
            model_dir=config.dataset_info,
            dataset='berson3d')
        test_dataset_module = test_dataset_module.data_processing(
            z_slices=z)
        test_dataset_module.file_pointer = config.test_dataset

    if force_meta:
        test_data_meta = np.load(
            os.path.join(
                config.tf_records,
                'class_weights_%s.npz' % force_meta))
    else:
        meta_path = os.path.join(
            config.tf_records,
            'class_weights_%s.npz' % test_dataset_module.output_name)
        if os.path.exists(meta_path):
            test_data_meta = np.load(meta_path)
        else:
            test_data_meta = None

    # Adjust params for these datasets
    if evaluate:
        params = experiment_params(
            test_name=test_dataset_module.name,
            test_shape=test_dataset_module.test_input_shape,
            z=z)
    else:
        params = experiment_params(
            train_name=train_dataset_module.name,
            test_name=test_dataset_module.name,
            train_shape=train_dataset_module.train_input_shape,
            test_shape=test_dataset_module.eval_input_shape,
            z=z)
    config = py_utils.add_to_config(
        d=params,
        config=config)
    if not evaluate:
        config.train_augmentations = check_augmentations(
            augmentations=config.train_augmentations,
            meta=train_data_meta)
    config.test_augmentations = check_augmentations(
        augmentations=config.test_augmentations,
        meta=test_data_meta)

    # Create labels
    dt = py_utils.get_dt_stamp()
    exp_label = '%s_%s_%s_%s' % (
        params['exp_label'],
        params['train_dataset'][0],
        params['test_dataset'][0],
        dt)
    summary_dir = os.path.join(config.summaries, exp_label)
    checkpoint_dir = os.path.join(config.checkpoints, exp_label)
    prediction_dir = os.path.join(config.predictions, exp_label)
    py_utils.make_dir(prediction_dir)
    return (
        config,
        exp_label,
        prediction_dir,
        checkpoint_dir,
        summary_dir,
        test_data_meta,
        test_dataset_module,
        train_dataset_module,
        train_data_meta)


def calculate_pr(labels, predictions, name, summation_method):
    """Calculate precision recall in an op that resets running tally."""
    pr, pr_update = tf.metrics.auc(
        labels=labels,
        predictions=predictions,
        curve='PR',
        name=name,
        summation_method=summation_method)
    running_vars = tf.get_collection(
        tf.GraphKeys.LOCAL_VARIABLES,
        scope=name)
    running_vars_initializer = tf.variables_initializer(
        var_list=running_vars)
    return pr, pr_update, running_vars_initializer


def convert_norm(meta):
    """Create the native normalization."""
    return {'normalize_volume': lambda x: (
        x - meta['min_val'].item()) / (
            meta['max_val'].item() - meta['min_val'].item())}


def convert_z(meta):
    """Create the native normalization."""
    return {'normalize_volume': lambda x: (
        x - meta['mean'].item()) / meta['std'].item()}


def uint8_normalization():
    """Return a x/255. normalization clipped to [0, 1]."""
    return {'normalize_volume': lambda x: tf.minimum(x / 255., 1)}


def check_augmentations(augmentations, meta):
    """Adjust augmentations based on keywords."""
    if isinstance(augmentations, list):
        for idx, augmentation in enumerate(augmentations):
            if 'min_max_native_normalization' in augmentation.keys():
                augmentations[idx] = convert_norm(meta)
            elif 'native_normalization_z' in augmentation.keys():
                augmentations[idx] = convert_z(meta)
            elif 'uint8_normalization' in augmentation.keys():
                augmentations[idx] = uint8_normalization()
    else:
        print 'WARNING: No normalization requested.'
    return augmentations


def prepare_data(
        tf_records,
        device,
        train_input_shape,
        train_label_shape,
        test_input_shape=None,
        test_label_shape=None,
        train_batch_size=None,
        test_batch_size=None,
        force_jk=False,
        dtype=tf.float32,
        evaluate=False):
    """Wrapper for tfrecord/placeholder data tensor creation."""
    with tf.device(device):
        train_images, train_labels = None, None
        if tf_records:
            raise NotImplementedError
        else:
            # Create data tensors
            assert (
                test_input_shape[1] / 8. == test_input_shape[
                    1] / 8),\
                'H/W must be divisible by 8.'
            with tf.device('/cpu:0'):
                if force_jk:
                    if not evaluate:
                        train_images = tf.placeholder(
                            dtype=dtype,
                            shape=[train_batch_size] +
                            train_input_shape,
                            name='train_images')
                        train_labels = tf.placeholder(
                            dtype=dtype,
                            shape=[train_batch_size] +
                            train_input_shape,
                            name='train_labels')
                    test_images = tf.placeholder(
                        dtype=dtype,
                        shape=[test_batch_size] +
                        test_input_shape,
                        name='test_images')
                    test_labels = tf.placeholder(
                        dtype=dtype,
                        shape=[test_batch_size] +
                        test_input_shape,
                        name='test_labels')
                else:
                    if not evaluate:
                        train_images = tf.placeholder(
                            dtype=dtype,
                            shape=[None] + train_input_shape,
                            name='train_images')
                        train_labels = tf.placeholder(
                            dtype=dtype,
                            shape=[None] + train_input_shape,
                            name='train_labels')
                    test_images = tf.placeholder(
                        dtype=dtype,
                        shape=[None] + test_input_shape,
                        name='test_images')
                    test_labels = tf.placeholder(
                        dtype=dtype,
                        shape=[None] + test_input_shape,
                        name='test_labels')
        return test_images, test_labels, train_images, train_labels


def initialize_tf(adabn, graph=None):
    # Initialize tf variables
    with graph.as_default():
        var_list = tf.global_variables()
        ada_initializer = None

        # If adaBN is requested, exclude running means from saver
        if adabn:  # hasattr(config, 'adabn') and config.adabn:
            moving_ops_names = ['moving_mean:', 'moving_variance:']
            var_list = [
                x for x in var_list
                if x.name.split('/')[-1].split(':')[0] + ':'
                not in moving_ops_names]
            moment_list = [
                x for x in tf.global_variables()
                if x.name.split('/')[-1].split(':')[0] + ':'
                in moving_ops_names]
            ada_initializer = tf.variables_initializer(
                var_list=moment_list)
    saver = tf.train.Saver(
        var_list=var_list)
    sess = tf.Session(graph=graph, config=tf.ConfigProto(
        allow_soft_placement=True))
    with sess.as_default():
        with graph.as_default():
            sess.run(
                tf.group(
                    tf.global_variables_initializer(),
                    tf.local_variables_initializer()))
    return sess, None, None, saver, ada_initializer


def evaluate_model(
        test=None,
        gpu_device='/gpu:0',
        cpu_device='/cpu:0',
        z=18,
        version='3d',
        build_model='None',
        experiment_params='experiment_params',
        checkpoint=None,
        full_volume=True,
        full_eval=False,
        force_meta=None,
        force_jk=False,
        bethge=None,
        adabn=False,
        return_sess=None,
        force_return_model=False,
        test_input_shape=False,
        test_label_shape=False,
        tf_dtype=tf.float32,
        tf_records=False):
    """Prepare a model for evaluation."""
    set_training = False
    if adabn:
        set_training = True
    model_graph = tf.Graph()
    with model_graph.as_default():
        test_images = tf.placeholder(
            dtype=tf_dtype,
            shape=[None] + test_input_shape,
            name='test_images')
        with tf.device(gpu_device):
            test_logits = build_model(
                data_tensor=test_images,
                reuse=None,
                training=set_training,
                output_channels=test_label_shape[-1])
            # Derive metrics
            test_scores = tf.sigmoid(test_logits)
    test_dict = {
        'test_logits': test_scores,
        'test_images': test_images
    }

    if force_return_model:
        return test_dict
    else:
        # Start evaluation
        sess, summary_op, summary_writer, saver, adabn_init = initialize_tf(
            adabn, model_graph)
        return training.evaluation_loop(
            sess=sess,
            test_data=test,
            saver=saver,
            checkpoint=checkpoint,
            full_volume=full_volume,
            return_sess=return_sess,
            test_dict=test_dict)


def train_model(
        train=None,
        test=None,
        row_id=None,
        gpu_device='/gpu:0',
        cpu_device='/cpu:0',
        z=18,
        train_input_shape=None,
        train_label_shape=None,
        test_input_shape=None,
        test_label_shape=None,
        version='3d',
        build_model=None,
        experiment_params=None,
        tf_records=False,
        checkpoint=None,
        weight_loss=True,
        overwrite_training_params=False,
        force_jk=False,
        use_bfloat16=False,
        wd=False,
        adabn=False,
        summary_dir=None,
        use_lms=False):
    """Run an experiment with hGRUs."""
    # Set up tensors
    (
        test_images,
        test_labels,
        train_images,
        train_labels) = prepare_data(
            tf_records=tf_records,
            device=cpu_device,
            train_input_shape=train_input_shape,
            train_label_shape=train_label_shape,
            test_input_shape=test_input_shape,
            test_label_shape=test_label_shape,
            force_jk=force_jk,
            evaluate=False)
    if use_bfloat16:
        train_images = tf.to_bfloat16(train_images)
        test_images = tf.to_bfloat16(test_images)

    # Build training and test models
    with tf.device(gpu_device):
        train_logits = build_model(
            data_tensor=train_images,
            reuse=None,
            training=True,
            output_channels=train_input_shape[-1])
        test_logits = build_model(
            data_tensor=test_images,
            reuse=tf.AUTO_REUSE,
            training=False,
            output_channels=train_input_shape[-1])

    # Derive loss
    if weight_loss:
        total_labels = np.prod(train_input_shape)
        count_pos = tf.reduce_sum(train_labels)
        count_neg = total_labels - count_pos
        beta = tf.cast(count_neg / (count_neg + count_pos), tf.float32)
        pos_weight = beta / (1 - beta)
        train_loss = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                targets=train_labels,
                logits=train_logits,
                pos_weight=pos_weight))
    else:
        train_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=train_labels,
                logits=train_logits))
    test_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=test_labels,
            logits=test_logits))
    if wd:
        train_loss += (WEIGHT_DECAY * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()
                if 'batch_normalization' not in v.name]))

    train_f1, train_precision, train_recall = f1_metric(
        y_true=train_labels, y_pred=tf.sigmoid(train_logits))
    test_f1, test_precision, test_recall = f1_metric(
        y_true=test_labels, y_pred=tf.sigmoid(test_logits))

    # Build optimizer
    lr = tf.placeholder(tf.float32, shape=[])
    train_op = optimizers.get_optimizer(
        loss=train_loss,
        lr=lr,
        optimizer='adam')

    # Create dictionaries of important training and test information
    train_dict = {
        'train_loss': train_loss,
        'train_images': train_images,
        'train_labels': train_labels,
        'train_op': train_op,
        'train_f1': train_f1,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_logits': train_logits
    }

    test_dict = {
        'test_loss': test_loss,
        'test_images': test_images,
        'test_labels': test_labels,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_logits': test_logits
    }

    # Count model parameters
    parameter_count = tf_fun.count_parameters(tf.trainable_variables())
    print 'Number of parameters in model: %s' % parameter_count
    sess, summary_op, summary_writer, saver, adabn_init = initialize_tf(
        adabn=adabn,
        summary_dir=summary_dir)

    # Start training loop
    if use_lms:
        from tensorflow.contrib.lms import LMS
        lms_model = LMS({'cnn'}, lb=3)  # Hardcoded model scope for now...
        lms_model.run(tf.get_default_graph())
    return sess, saver, train_dict, test_dict

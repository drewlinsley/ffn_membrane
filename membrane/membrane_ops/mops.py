
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
        config,
        tf_records,
        device,
        test_dataset_module,
        train_dataset_module=None,
        force_jk=False,
        evaluate=False):
    """Wrapper for tfrecord/placeholder data tensor creation."""
    with tf.device(device):
        train_images, train_labels = None, None
        if tf_records:
            if not evaluate:
                train_dataset = os.path.join(
                    config.tf_records,
                    '%s_train.tfrecords' % train_dataset_module.output_name)
                train_images, train_labels = data_loader.inputs(
                    dataset=train_dataset,
                    batch_size=config['train_batch_size'],
                    input_shape=config['train_input_shape'],
                    label_shape=config['train_label_shape'],
                    tf_dict=train_dataset_module.tf_dict,
                    data_augmentations=config['train_augmentations'],
                    num_epochs=config['epochs'],
                    tf_reader_settings=train_dataset_module.tf_reader,
                    shuffle=config['shuffle_train'])
            test_dataset = os.path.join(
                config.tf_records,
                '%s_test.tfrecords' % test_dataset_module.output_name)
            test_images, test_labels = data_loader.inputs(
                dataset=test_dataset,
                batch_size=config['test_batch_size'],
                input_shape=config['test_input_shape'],
                label_shape=config['test_label_shape'],
                tf_dict=test_dataset_module.tf_dict,
                data_augmentations=config['test_augmentations'],
                num_epochs=config['epochs'],
                tf_reader_settings=test_dataset_module.tf_reader,
                shuffle=config['shuffle_test'])
        else:
            # Create data tensors
            with tf.device('/cpu:0'):
                assert (
                    config.test_input_shape[1] / 8. == config.test_input_shape[
                        1] / 8),\
                    'H/W must be divisible by 8.'
                if force_jk:
                    if not evaluate:
                        train_images = tf.placeholder(
                            dtype=config.tf_dtype,
                            shape=[config['train_batch_size']] +
                            config.train_input_shape,
                            name='train_images')
                        train_labels = tf.placeholder(
                            dtype=config.tf_dtype,
                            shape=[config['train_batch_size']] +
                            config.train_label_shape,
                            name='train_labels')
                    test_images = tf.placeholder(
                        dtype=config.tf_dtype,
                        shape=[config['test_batch_size']] +
                        config.test_input_shape,
                        name='test_images')
                    test_labels = tf.placeholder(
                        dtype=config.tf_dtype,
                        shape=[config['test_batch_size']] +
                        config.test_label_shape,
                        name='test_labels')
                else:
                    if not evaluate:
                        train_images = tf.placeholder(
                            dtype=config.tf_dtype,
                            shape=[None] + config.train_input_shape,
                            name='train_images')
                        train_labels = tf.placeholder(
                            dtype=config.tf_dtype,
                            shape=[None] + config.train_label_shape,
                            name='train_labels')
                    test_images = tf.placeholder(
                        dtype=config.tf_dtype,
                        shape=[None] + config.test_input_shape,
                        name='test_images')
                    test_labels = tf.placeholder(
                        dtype=config.tf_dtype,
                        shape=[None] + config.test_label_shape,
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

    # Start evaluation
    sess, summary_op, summary_writer, saver, adabn_init = initialize_tf(adabn, model_graph)
    return training.evaluation_loop(
        sess=sess,
        test_data=test,
        saver=saver,
        checkpoint=checkpoint,
        full_volume=full_volume,
        test_dict=test_dict)


def train_model(
        train=None,
        test=None,
        row_id=None,
        gpu_device='/gpu:0',
        cpu_device='/cpu:0',
        z=18,
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
        use_lms=False):
    """Run an experiment with hGRUs."""
    # Set up tensors
    (
        config,
        exp_label,
        prediction_dir,
        checkpoint_dir,
        summary_dir,
        test_data_meta,
        test_dataset_module,
        train_dataset_module,
        train_data_meta) = configure_model(
            train=train,
            test=test,
            row_id=row_id,
            gpu_device=gpu_device,
            z=z,
            version=version,
            build_model=build_model,
            experiment_params=experiment_params,
            evaluate=False)
    if overwrite_training_params:
        config = tf_fun.update_config(overwrite_training_params, config)
    config.ds_name = {
        'train': train,
        'test': test
    }

    (
        test_images,
        test_labels,
        train_images,
        train_labels) = prepare_data(
            config=config,
            tf_records=tf_records,
            device=cpu_device,
            test_dataset_module=test_dataset_module,
            train_dataset_module=train_dataset_module,
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
            output_channels=config.train_label_shape[-1])
        test_logits = build_model(
            data_tensor=test_images,
            reuse=tf.AUTO_REUSE,
            training=False,
            output_channels=config.test_label_shape[-1])

    if use_bfloat16:
        train_logits = tf.cast(train_logits, experiment_params.tf_dtype)
        test_logits = tf.cast(test_logits, experiment_params.tf_dtype)

    # Derive loss
    if weight_loss:
        assert train_data_meta is not None, 'Could not find a train_data_meta'
        pos_weight = train_data_meta['weights']
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
        WEIGHT_DECAY = 1e-4
        train_loss += (WEIGHT_DECAY * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()
            if 'batch_normalization' not in v.name]))

    # Derive metrics
    train_scores = tf.reduce_mean(
        tf.sigmoid(train_logits[:, :, :, :, :3]),
        axis=-1)  # config['gt_idx']])
    test_scores = tf.reduce_mean(
        tf.sigmoid(test_logits[:, :, :, :, :3]),
        axis=-1)  # config['gt_idx']])
    train_gt = tf.cast(
        tf.greater(
            tf.reduce_mean(train_labels[:, :, :, :, :3], axis=-1), 0.5),
        tf.int32)  # config['gt_idx']]
    test_gt = tf.cast(
        tf.greater(
            tf.reduce_mean(test_labels[:, :, :, :, :3], axis=-1), 0.5),
        tf.int32)  # config['gt_idx']]
    try:
        train_pr, train_pr_update, train_pr_init = calculate_pr(
            labels=train_gt,
            predictions=train_scores,
            summation_method='careful_interpolation',
            name='train_pr')
        test_pr, test_pr_update, test_pr_init = calculate_pr(
            labels=test_gt,
            predictions=test_scores,
            summation_method='careful_interpolation',
            name='test_pr')
    except Exception:
        print 'Failed to use careful_interpolation'
        train_pr, train_pr_update, train_pr_init = calculate_pr(
            labels=train_gt,
            predictions=train_scores,
            summation_method='trapezoidal',
            name='train_pr')
        test_pr, test_pr_update, test_pr_init = calculate_pr(
            labels=test_gt,
            predictions=test_scores,
            summation_method='trapezoidal',
            name='test_pr')
    train_metrics = {
        'train_pr': train_pr,
        'train_cce': train_loss
    }
    test_metrics = {
        'test_pr': test_pr,
        'test_cce': test_loss
    }
    for k, v in train_metrics.iteritems():
        if 'update' not in k:
            tf.summary.scalar(k, v)
    for k, v in test_metrics.iteritems():
        if 'update' not in k:
            tf.summary.scalar(k, v)

    # Build optimizer
    lr = tf.placeholder(tf.float32, shape=[])
    train_op = optimizers.get_optimizer(
        loss=train_loss,
        lr=lr,
        optimizer=config['optimizer'])

    # Create dictionaries of important training and test information
    train_dict = {
        'train_loss': train_loss,
        'train_images': train_images,
        'train_labels': train_labels,
        'train_op': train_op,
        'train_pr_update': train_pr_update,
        'train_logits': train_scores
    }

    test_dict = {
        'test_loss': test_loss,
        'test_images': test_images,
        'test_labels': test_labels,
        'test_pr_update': test_pr_update,
        'test_logits': test_scores
    }
    train_metrics = {
        'train_pr': train_pr,
    }
    test_metrics = {
        'test_pr': test_pr,
    }
    reset_metrics = {
        'train_pr_init': train_pr_init,
        'test_pr_init': test_pr_init,
    }

    # Count model parameters
    parameter_count = tf_fun.count_parameters(tf.trainable_variables())
    print 'Number of parameters in model: %s' % parameter_count

    # Create datastructure for saving data
    ds = data_structure.data(
        train_batch_size=config.train_batch_size,
        test_batch_size=config.test_batch_size,
        test_iters=config.test_iters,
        shuffle_train=config.shuffle_train,
        shuffle_test=config.shuffle_test,
        lr=config.lr,
        training_routine=config.training_routine,
        loss_function=config.loss_function,
        optimizer=config.optimizer,
        model_name=config.exp_label,
        train_dataset=config.train_dataset,
        test_dataset=config.test_dataset,
        output_directory=config.results,
        prediction_directory=prediction_dir,
        summary_dir=summary_dir,
        checkpoint_dir=checkpoint_dir,
        parameter_count=parameter_count,
        exp_label=exp_label)
    sess, summary_op, summary_writer, saver, adabn_init = initialize_tf(
        config=config,
        summary_dir=summary_dir)

    # Start training loop
    if use_lms:
        from tensorflow.contrib.lms import LMS
        lms_model = LMS({'cnn'}, lb=3)  # Hardcoded model scope for now...
        lms_model.run(tf.get_default_graph())
    if tf_records:
        # Coordinate for tfrecords
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        training_tf.training_loop(
            config=config,
            sess=sess,
            summary_op=summary_op,
            summary_writer=summary_writer,
            saver=saver,
            summary_dir=summary_dir,
            checkpoint_dir=checkpoint_dir,
            prediction_dir=prediction_dir,
            train_dict=train_dict,
            test_dict=test_dict,
            exp_label=config.exp_label,
            lr=lr,
            row_id=row_id,
            data_structure=ds,
            coord=coord,
            threads=threads,
            reset_metrics=reset_metrics,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            checkpoint=checkpoint,
            top_test=config['top_test'])
    else:
        training.training_loop(
            config=config,
            sess=sess,
            summary_op=summary_op,
            summary_writer=summary_writer,
            saver=saver,
            summary_dir=summary_dir,
            checkpoint_dir=checkpoint_dir,
            prediction_dir=prediction_dir,
            train_dict=train_dict,
            test_dict=test_dict,
            train_dataset_module=train_dataset_module,
            test_dataset_module=test_dataset_module,
            exp_label=config.exp_label,
            lr=lr,
            row_id=row_id,
            data_structure=ds,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            reset_metrics=reset_metrics,
            checkpoint=checkpoint,
            top_test=config['top_test'])


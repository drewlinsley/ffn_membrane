import os
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
from membrane.utils import logger
from membrane.membrane_ops import data_utilities
from membrane.membrane_ops import tf_fun
# from db import db
# from memory_profiler import profile

# try:
#     from bsds import evaluate_boundaries
#     bsds_metric = False  # True
# except Exception:
#     print 'Failed to import BSDS metrics.'
#     bsds_metric = False
FORGETTING_VOLUMES = [
    '/media/data_cifs/connectomics/datasets/berson_0.npz',
    '/media/data_cifs/connectomics/datasets/fib25_0.npz',
]


def load_data(data, shape=[128, 128, 128]):
    """Load numpy or raw files."""
    if 'npy' in data or 'npz' in data:
        return np.load(data)
    else:
        # Load a raw file
        if isinstance(data, list):
            rs = []
            for d in data:
                rs += [np.fromfile(d, dtype='uint8').reshape(shape)]
            rs = np.asarray(rs)
            # Reconstruct order from filenames
            cube_len = int(float(len(data)) ** (1 / 3.))
            data_coors = tf_fun.strip_coors(data)
            recoded_coors = np.asarray(
                (data_coors - data_coors.min(0)))  # [:, [1, 0, 2]]
            edge = shape[0]
            recoded_boundaries = range(
                edge,
                edge * cube_len,
                edge) + [edge * cube_len]
            supervolume = np.zeros((np.asarray(shape) * cube_len))
            history = []
            for x, x_boundary in enumerate(recoded_boundaries):
                for y, y_boundary in enumerate(recoded_boundaries):
                    for z, z_boundary in enumerate(recoded_boundaries):
                        data_idx = np.all(
                            recoded_coors == [x, y, z],
                            axis=1)
                        history += [[x, y, z]]
                        supervolume[
                            z_boundary - edge: z_boundary,
                            y_boundary - edge: y_boundary,
                            x_boundary - edge: x_boundary,
                        ] = rs[data_idx].squeeze()
            return {
                'volume': supervolume,
                'label': np.zeros_like(supervolume).astype(np.int32)
            }
        else:
            supervolume = np.fromfile(data, dtype='uint8').reshape(shape)
            return {
                'volume': supervolume,
                'label': np.zeros_like(supervolume).astype(np.int32)
            }


def cv_split(volume, config, key):
    """Split volume according to cv."""
    if hasattr(config, 'reduction') and key == 'train':
        ds_name = config.ds_name[key]
        pivots = tf_fun.finetune_splits(
            dataset=ds_name, reduction=config.reduction) 
        starts = [x[0] if isinstance(x, tuple) else 0 for x in pivots]
        ends = [x[1] if isinstance(x, tuple) else x for x in pivots]
        volume = volume[
            starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
        hw_check = np.asarray(volume.shape[1:]) < config.train_input_shape[1:3]
        if np.any(hw_check):
            for idx, hw in enumerate(hw_check):
                if idx == 0 and hw:
                    # Pad H
                    offset = config.train_input_shape[1] - volume.shape[1] + 1
                    zero_slab = np.zeros((volume.shape[0], offset, volume.shape[2]))
                    volume = np.concatenate((volume, zero_slab), axis=1)
                elif idx == 1 and hw:
                    # Pad W
                    offset = config.train_input_shape[2] - volume.shape[2] + 1
                    zero_slab = np.zeros((volume.shape[0], volume.shape[1], offset))
                    volume = np.concatenate((volume, zero_slab), axis=2)
    else:
        cv_split = config.cross_val[key]
        vshape = volume.shape
        start = int(vshape[0] * (cv_split[0] / 100.))
        end = int(vshape[0] * (cv_split[1] / 100.))
        volume = volume[np.arange(start, end)]
    return volume


def get_z_idx(volume_shape, stride, num_slices, include_if_less=False):
    """Get strided z-axis index."""
    z_idx_start = np.arange(
        0,
        volume_shape - stride,
        stride)
    if not len(z_idx_start):
        z_idx_start = np.asarray([0])
    z_idx_end = z_idx_start + stride
    z_idx_start = np.concatenate((
        z_idx_start,
        np.asarray(z_idx_end[-1]).ravel()))
    if not include_if_less:
        z_idx_start = z_idx_start[z_idx_start <= num_slices]  # Untested
    return z_idx_start, z_idx_end


def prepare_idx(z_idx_start, z_idx_end, shuffle, batch_size):
    """Prepare index for training or testing."""
    if shuffle:
        it_idx = np.random.permutation(len(z_idx_start))
    else:
        it_idx = np.arange(len(z_idx_start))
    ss = z_idx_start[it_idx]
    se = z_idx_end  # [it_idx] NOT USING SE

    # Trim and reshape for batches
    batches = np.floor(float(len(ss)) / batch_size).astype(int)
    batch_cut = int(batches * batch_size)
    ss = ss[:batch_cut]
    # se = se[:batch_cut]
    ss = ss.reshape(batches, batch_size)
    # se = se.reshape(batches, batch_size)
    return ss, se, batches


def sample_data(volume, label, shape, ss, z, dtype, config, fold):
    """Sample from volume and label according to ss."""
    batch_volume = []
    batch_label = []
    for s in ss:
        if s + z > shape[0]:
            if shape[1] != volume.shape[1]:
                shape[1:] = volume.shape[1:3]
            diff = abs(shape[0] - z - s)
            pad_volume = volume[s:shape[0]]
            pad_label = label[s:shape[0]]
            pad_volume = np.concatenate((
                pad_volume,
                np.zeros([diff] + list(shape[1:3]))), axis=0)
            pad_label = np.concatenate((
                pad_label,
                np.zeros([diff] + list(pad_label.shape[1:]))), axis=0)
            batch_volume += [pad_volume]
            batch_label += [pad_label]
        else:
            batch_volume += [volume[s:s + z]]
            batch_label += [label[s:s + z]]
    batch_volume = np.stack(batch_volume, axis=0).astype(dtype)
    batch_label = np.stack(batch_label, axis=0).astype(dtype)
    vshape, lshape = len(batch_volume.shape), len(batch_label.shape)
    if vshape == 4:
        batch_volume = np.expand_dims(batch_volume, axis=-1)
    elif vshape == 5:
        pass
    else:
        raise RuntimeError('Something is wrong with your volume size: %s' % (
            batch_volume.shape))
    if lshape == 4:
        batch_label = np.expand_dims(batch_label, axis=-1)
    elif lshape == 5:
        pass
    else:
        raise RuntimeError('Something is wrong with your label size: %s' % (
            batch_label.shape))
    augmentation_list = getattr(config, '%s_augmentations' % fold)
    if not isinstance(augmentation_list, list):
        augmentation_list = [augmentation_list]
    input_shape = getattr(config, '%s_input_shape' % fold)
    label_shape = getattr(config, '%s_label_shape' % fold)
    if len(augmentation_list):
        batch_volume, batch_label = data_utilities.apply_augmentations(
            volume=batch_volume,
            label=batch_label,
            input_shape=input_shape,
            label_shape=label_shape,
            augmentations=augmentation_list)
    return batch_volume, batch_label


def run_train_step(
        idx,
        sess,
        train_volume,
        train_label,
        train_ss,
        config,
        train_shape,
        train_dict,
        train_metrics,
        lr,
        reset_metrics,
        dtype,
        it_lr,
        step,
        log,
        data_structure):
    """Run a step of training."""
    train_batch_volumes, train_batch_labels = sample_data(
        volume=train_volume,
        label=train_label,
        ss=train_ss[idx],
        z=config.train_input_shape[0],
        shape=train_shape,
        dtype=config.np_dtype,
        config=config,
        fold='train')
    start_time = time.time()
    if (train_batch_volumes.shape[2] == int(
            train_dict['train_images'].get_shape()[2])):
        feed_dict = {
            train_dict['train_images']: train_batch_volumes,
            train_dict['train_labels']: train_batch_labels,
            lr: it_lr
        }
        # Reset streaming ops
        sess.run(reset_metrics['train_pr_init'])

        # Run sess
        it_train_dict = sess.run(
            train_dict,
            feed_dict=feed_dict)
        it_train_metrics = sess.run(
            train_metrics,
            feed_dict=feed_dict)
        duration = time.time() - start_time
        train_loss = np.asarray(
            it_train_dict['train_loss']).astype(dtype)
        train_pr = np.asarray(
            it_train_metrics['train_pr']).astype(dtype)
        # train_losses += [train_loss]
        # train_prs += [train_pr]
        # train_accs += [it_train_dict['train_accuracy']]
        # train_arand += [it_train_dict['train_arand']]
        # timesteps += [duration]
        try:
            data_structure.update_training(
                train_pr=train_pr,
                # train_accuracy=it_train_dict['train_accuracy'],
                # train_arand=it_train_dict['train_arand'],
                train_loss=train_loss,
                train_step=step)
            data_structure.save()
        except Exception as e:
            log.warning('Failed to update saver class: %s' % e)
        # End iteration
        step += 1
    else:
        print 'Missed a crop req? Im shape is different than expected.'
    return step, duration, train_pr, train_loss


def test_evaluation(
        sess,
        test_dict,
        test_volume,
        test_shape,
        test_z_idx_start,
        test_z_idx_end,
        dtype,
        log,
        adabn_init=None):
    """Evaluate test data."""
    # Prepare test idx for current epoch
    test_ss, test_se, test_batches = prepare_idx(
        test_z_idx_start,
        test_z_idx_end,
        config.shuffle_test,
        config.test_batch_size)

    # Check that the dataset will fit with the test_ss
    if not len(test_ss):
        test_ss, test_batches = [[0]], 1
    elif (
        (test_ss[-1] + config.test_input_shape[0]) > test_volume.shape[0] and
            'cremi' not in config.test_dataset):
        test_ss = test_ss[:-1]
        test_se = test_se[:-1]
        test_batches -= 1

    it_test_loss = []
    # it_test_arand = []
    # it_test_acc = []
    it_test_scores = []
    it_test_labels = []
    it_test_volumes = []
    it_test_pr = []
    # it_test_bsds = []
    for num_vals in range(test_batches):
        log.info('Testing %s...' % num_vals)
        test_batch_volumes, test_batch_labels = sample_data(
            volume=test_volume,
            label=test_label,
            ss=test_ss[num_vals],
            z=config.test_input_shape[0],
            shape=test_shape,
            dtype=config.np_dtype,
            config=config,
            fold='test')

        # Test accuracy as the average of n batches
        feed_dict = {
            test_dict['test_images']: test_batch_volumes,
        }

        sess.run(reset_metrics['test_pr_init'])
        it_test_dict = sess.run(
            test_dict,
            feed_dict=feed_dict)
        it_test_scores += [it_test_dict['test_logits']]
    return _, _, _, it_test_scores, _


def test_wrapper(
        test_volume,
        config,
        sess,
        test_dict,
        test_z_idx_start,
        test_z_idx_end,
        dtype,
        log):
    """Wrapper for running test ops."""
    if test_dataset_module.file_pointer == 'forgetting':
        test_lo = []
        test_pr = []
        it_test_volumes = []
        it_test_scores = []
        it_test_labels = []
        for fidx, (sel_vol, sel_lab) in enumerate(
                zip(test_volume, test_label)):
            print 'Evaluating volume %s/%s' % (
                fidx, len(FORGETTING_VOLUMES))
            test_shape = sel_vol.shape
            (
                f_test_lo,
                f_test_pr,
                f_it_test_volumes,
                f_it_test_scores,
                f_it_test_labels) = test_evaluation(
                config=config,
                sess=sess,
                test_dict=test_dict,
                test_volume=sel_vol,
                test_label=sel_lab,
                test_metrics=test_metrics,
                test_shape=test_shape,
                test_z_idx_start=test_z_idx_start[fidx],
                test_z_idx_end=test_z_idx_end[fidx],
                reset_metrics=reset_metrics,
                dtype=dtype,
                log=log)
            test_lo += [f_test_lo]
            test_pr += [f_test_pr]
            it_test_volumes += [f_it_test_volumes]
            it_test_scores += [f_it_test_scores]
            it_test_labels += [f_it_test_labels]

    else:
        test_shape = test_volume.shape
        (
            test_lo,
            test_pr,
            it_test_volumes,
            it_test_scores,
            it_test_labels) = test_evaluation(
            config=config,
            sess=sess,
            test_dict=test_dict,
            test_volume=test_volume,
            test_label=test_label,
            test_metrics=test_metrics,
            test_shape=test_shape,
            test_z_idx_start=test_z_idx_start,
            test_z_idx_end=test_z_idx_end,
            reset_metrics=reset_metrics,
            dtype=dtype,
            log=log)
    return (
        test_lo,
        test_pr,
        it_test_volumes,
        it_test_scores,
        it_test_labels)


# @profile
def training_loop(
        config,
        sess,
        summary_op,
        summary_writer,
        saver,
        summary_dir,
        checkpoint_dir,
        prediction_dir,
        train_dict,
        test_dict,
        exp_label,
        train_dataset_module,
        test_dataset_module,
        lr,
        row_id,
        data_structure,
        reset_metrics,
        checkpoint,
        train_metrics,
        test_metrics,
        log=None,
        dtype=np.float16,
        transpose=False,  # (2, 0, 1),
        top_test=5):
    """Run the model training loop."""
    if log is None:
        log = logger.get(
            os.path.join(config.log_dir, summary_dir.split(os.path.sep)[-1]))
    if checkpoint is not None:
        saver.restore(sess, checkpoint)
        log.info('Restoring checkpoint %s' % checkpoint)
    step = 0
    test_losses = []

    # Load train and test volumes into memory
    train_data = np.load(train_dataset_module.file_pointer)
    if test_dataset_module.file_pointer == 'forgetting':
        test_data = [np.load(x) for x in FORGETTING_VOLUMES]
    else:
        test_data = np.load(test_dataset_module.file_pointer)
    log.info('Loading data.')
    if test_dataset_module.file_pointer == 'forgetting':
        if transpose:
            train_volume = train_data['volume'].transpose(transpose)
            train_label = train_data['label'].transpose(transpose)
        else:
            train_volume = train_data['volume']
            train_label = train_data['label']
        train_volume = cv_split(
            train_volume, config, key='train')
        # .cross_val['train'], config, ds_name=config.train_name)
        train_label = cv_split(
            train_label, config, key='train')
        if config.affinity > 1:
            train_label = data_utilities.derive_affinities(
                affinity=config.affinity,
                label_volume=train_label)

        test_vols, test_labs = [], []
        for vol in test_data:
            if transpose:
                tv = vol['volume'].transpose(transpose)
                tl = vol['label'].transpose(transpose)
            else:
                tv = vol['volume']
                tl = vol['label']
            # Split into CV folds
            test_volume = cv_split(
                tv, config, key='test')
            test_label = cv_split(
                tl, config, key='test')

            # Prepare affinity label volumes if requested
            if config.affinity > 1:
                tl = data_utilities.derive_affinities(
                    affinity=config.affinity,
                    label_volume=tl)
            test_vols += [tv]
            test_labs += [tl]
        test_volume = test_vols
        test_label = test_labs
    else:
        if transpose:
            train_volume = train_data['volume'].transpose(transpose)
            train_label = train_data['label'].transpose(transpose)
            test_volume = test_data['volume'].transpose(transpose)
            test_label = test_data['label'].transpose(transpose)
        else:
            train_volume = train_data['volume']
            train_label = train_data['label']
            test_volume = test_data['volume']
            test_label = test_data['label']

        # Split into CV folds
        train_volume = cv_split(
            train_volume, config, key='train')
        # .cross_val['train'], config, ds_name=config.train_name)
        train_label = cv_split(
            train_label, config, key='train')
        # .cross_val['train'], config, ds_name=config.train_name)
        test_volume = cv_split(
            test_volume, config, key='test')
        # .cross_val['test'], config, ds_name=config.test_name)
        test_label = cv_split(
            test_label, config, key='test')
        # .cross_val['test'], config, ds_name=config.test_name)

        # Prepare affinity label volumes if requested
        if config.affinity > 1:
            train_label = data_utilities.derive_affinities(
                affinity=config.affinity,
                label_volume=train_label)
            test_label = data_utilities.derive_affinities(
                affinity=config.affinity,
                label_volume=test_label)

    # Derive indices for train/test data
    train_shape = train_volume.shape
    train_z_idx_start, train_z_idx_end = get_z_idx(
        volume_shape=train_shape[0],
        stride=config.train_stride[-1],
        num_slices=train_volume.shape[0] - config.train_input_shape[0],
        include_if_less=False)
    if test_dataset_module.file_pointer == 'forgetting':
        test_z_idx_start, test_z_idx_end = [], []
        for vol, lab in zip(test_vols, test_labs):
            test_shape = vol.shape
            idx_start, idx_end = get_z_idx(
                volume_shape=test_shape[0],
                stride=config.test_input_shape[0] // 2,  # half depth
                num_slices=vol.shape[0] - config.test_input_shape[0],
                include_if_less=False)
            test_z_idx_start += [idx_start]
            test_z_idx_end += [idx_end]
    else:
        test_shape = test_volume.shape
        test_z_idx_start, test_z_idx_end = get_z_idx(
            volume_shape=test_shape[0],
            stride=config.test_input_shape[0] // 2,  # half depth
            num_slices=test_volume.shape[0] - config.test_input_shape[0],
            include_if_less=False)

    # Set starting lr
    it_lr = config.lr
    lr_info = None

    # Update DB
    if row_id is not None:
        db.update_results(results=summary_dir, row_id=row_id)

    # Start loop
    em = None
    test_perf = np.ones(top_test) * np.inf
    # try:
    if checkpoint is not None:
        # Get 0-shot perf
        log.info('Calculating 0-shot baselines on original and new datasets.')
        (
            test_lo,
            test_pr,
            it_test_volumes,
            it_test_scores,
            it_test_labels) = test_wrapper(
                test_dataset_module=test_dataset_module,
                test_volume=test_volume,
                test_label=test_label,
                config=config,
                sess=sess,
                test_dict=test_dict,
                test_metrics=test_metrics,
                test_z_idx_start=test_z_idx_start,
                test_z_idx_end=test_z_idx_end,
                reset_metrics=reset_metrics,
                dtype=dtype,
                log=log)
        test_losses += [test_lo]

        # Update data structure
        try:
            data_structure.update_test(
                test_pr=test_pr,
                test_loss=test_lo,
                test_step=step,
                test_lr_info=lr_info,
                test_lr=it_lr)
            data_structure.save()
        except Exception as e:
            log.warning('Failed to update saver class: %s' % e)

        # 0-shot test accuracy
        format_str = ('Test pr = %s, Test loss = %s | logdir = %s')
        log.info(format_str % (
            test_pr,
            test_lo,
            summary_dir))

    # Check for request to stop training by # of steps
    if hasattr(config, 'max_steps'):
        max_steps = config.max_steps
    else:
        max_steps = np.inf

    for epoch in range(config.epochs):
        # Prepare train idx for current epoch
        log.info('Starting epoch %s/%s' % (epoch, config.epochs))
        train_ss, _, train_batches = prepare_idx(
            train_z_idx_start,
            train_z_idx_end,
            config.shuffle_train,
            config.train_batch_size)
        for idx in range(train_batches):
            # Train batch
            step, duration, train_pr, train_loss = run_train_step(
                idx=idx,
                sess=sess,
                train_volume=train_volume,
                train_label=train_label,
                train_ss=train_ss,
                config=config,
                train_shape=train_shape,
                train_dict=train_dict,
                train_metrics=train_metrics,
                lr=lr,
                reset_metrics=reset_metrics,
                dtype=dtype,
                it_lr=it_lr,
                step=step,
                log=log,
                data_structure=data_structure)

            if step % config.test_iters == 0:
                (
                    test_lo,
                    test_pr,
                    it_test_volumes,
                    it_test_scores,
                    it_test_labels) = test_wrapper(
                        test_dataset_module=test_dataset_module,
                        test_volume=test_volume,
                        test_label=test_label,
                        config=config,
                        sess=sess,
                        test_dict=test_dict,
                        test_metrics=test_metrics,
                        test_z_idx_start=test_z_idx_start,
                        test_z_idx_end=test_z_idx_end,
                        reset_metrics=reset_metrics,
                        dtype=dtype,
                        log=log)
                test_losses += [test_lo]

                # Update data structure
                try:
                    data_structure.update_test(
                        test_pr=test_pr,
                        test_loss=test_lo,
                        test_step=step,
                        test_lr_info=lr_info,
                        test_lr=it_lr)
                    data_structure.save()
                except Exception as e:
                    log.warning('Failed to update saver class: %s' % e)

                # Update data structure
                try:
                    if row_id is not None:
                        db.update_step(step=step, row_id=row_id)
                except Exception as e:
                    log.warning('Failed to update step count: %s' % e)

                # Save checkpoint
                ckpt_path = os.path.join(
                    checkpoint_dir,
                    'model_%s.ckpt' % step)
                try:
                    test_check = np.where(test_lo < test_perf)[0]
                    if len(test_check):
                        saver.save(
                            sess,
                            ckpt_path,
                            global_step=step)
                        if len(test_check):
                            test_check = test_check[0]
                        test_perf[test_check] = test_lo
                        log.info('Saved checkpoint to: %s' % ckpt_path)

                        # Save predictions
                        pred_path = os.path.join(
                            prediction_dir,
                            'model_%s' % step)
                        np.savez(
                            pred_path,
                            volumes=it_test_volumes,
                            predictions=it_test_scores,
                            labels=it_test_labels)
                except Exception as e:
                    log.info('Failed to save checkpoint.')

                # Update LR
                if not test_dataset_module.file_pointer == 'forgetting':
                    it_lr, lr_info = tf_fun.update_lr(
                        it_lr=it_lr,
                        test_losses=test_losses,
                        alg=config.training_routine,
                        lr_info=lr_info)

                # Summaries
                # summary_str = sess.run(summary_op)
                # summary_writer.add_summary(summary_str, step)

                # Training status and test accuracy
                format_str = (
                    '%s: step %d, train loss = %.2f (%.1f examples/sec; '
                    '%.3f sec/batch)'
                    ' | Train pr = %s, Test pr = %s, '
                    'Test loss = %s | logdir = %s')
                log.info(format_str % (
                    datetime.now(),
                    step,
                    train_loss,
                    config.train_batch_size / duration,
                    float(duration),
                    train_pr,
                    test_pr,
                    test_lo,
                    summary_dir))
                # Reset streaming ops
                sess.run(reset_metrics['test_pr_init'])
            else:
                # Training status
                format_str = (
                    '%s: step %d, loss = %.5f (%.1f examples/sec; '
                    '%.3f sec/batch) | '
                    'Training pr = %s')
                log.info(format_str % (
                    datetime.now(),
                    step,
                    train_loss,
                    config.train_batch_size / duration,
                    float(duration),
                    # it_train_dict['train_accuracy'],
                    train_pr))
            if step >= max_steps:
                # Reached max requested step of training
                break
        if step >= max_steps:
            break

    # except Exception as em:
    #     log.warning('Failed training: %s' % em)
    #     if row_id is not None:
    #         db.update_error(error=True, row_id=row_id)
    try:
        data_structure.update_error(msg=em)
        data_structure.save()
    except Exception as e:
        log.warning('Failed to update saver class: %s' % e)
    log.info('Done training for %d epochs, %d steps.' % (config.epochs, step))
    log.info('Saved to: %s' % checkpoint_dir)
    sess.close()
    return


def evaluation_loop(
        sess,
        saver,
        test_data,
        checkpoint,
        test_dict,
        full_volume=True,
        adabn_init=None,
        dtype=np.float32,
        return_sess=None,
        transpose=False,  # (2, 0, 1),
        force_stride=True,  # 50% stride
        use_padding=False,
        log=None):
    """Run the model training loop."""
    if log is None:
        log = logger.get("/media/data_cifs_lrs/projects/prj_connectomics/connectomics_data/segmentation_logs/log")

    # Restore checkpoint
    saver.restore(sess, checkpoint)

    # Determine padding
    if use_padding:
        raise NotImplementedError
        # test_shape = test_label.shape
        # pad_diff = int(
        #     np.round(
        #         test_volume.shape[0] / float(
        #             config.test_input_shape[0])
        #         ) * config.test_input_shape[0] - test_shape[0])
        # test_pad = np.zeros((pad_diff, test_shape[1], test_shape[2]))
        # test_volume = np.concatenate((test_volume, test_pad), axis=0)
    td_shape = test_data.shape
    td_shape_len = len(td_shape)
    if td_shape_len == 3:
        test_data = test_data[None, ..., None]
    elif td_shape_len == 4:
        test_data = test_data[..., None]
    elif td_shape_len == 5:
        pass
    else:
        raise NotImplementedError(td_shape_len)

    if td_shape_len > 3:
        # Loop through batch
        it_test_scores = []
        for td in tqdm(
                test_data, total=len(test_data), desc='Processing membranes'):
            td = td[None]
            feed_dict = {
                test_dict['test_images']: td,
            }
            it_test_dict = sess.run(
                test_dict,
                feed_dict=feed_dict)
            it_test_scores += [it_test_dict['test_logits']]
    else:
        feed_dict = {
            test_dict['test_images']: test_data,
        }
        it_test_dict = sess.run(
            test_dict,
            feed_dict=feed_dict)
        it_test_scores = it_test_dict['test_logits']
    if return_sess:
        return it_test_scores, sess, test_dict
    else:
        return it_test_scores


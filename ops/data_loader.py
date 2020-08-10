import math
import numpy as np
import tensorflow as tf
from scipy import stats
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops


def repeat_elements(x, rep, axis):
    '''Repeats the elements of a tensor along an axis, like np.repeat
    If x has shape (s1, s2, s3) and axis=1, the output
    will have shape (s1, s2 * rep, s3)
    This function is taken from keras backend
    '''
    x_shape = x.get_shape().as_list()
    splits = tf.split(axis=axis, num_or_size_splits=x_shape[axis], value=x)
    x_rep = [s for s in splits for i in range(rep)]
    return tf.concat(axis=axis, values=x_rep)


def resize_image_label(im, model_input_image_size, f='bilinear'):
    """Resize images filter."""
    if f == 'bilinear':
        res_fun = tf.image.resize_images
    elif f == 'nearest':
        res_fun = tf.image.resize_nearest_neighbor
    elif f == 'bicubic':
        res_fun = tf.image.resize_bicubic
    elif f == 'area':
        res_fun = tf.image.resize_area
    else:
        raise NotImplementedError
    if len(im.get_shape()) > 3:
        # Spatiotemporal image set.
        nt = int(im.get_shape()[0])
        sims = tf.split(im, nt)
        for idx in range(len(sims)):
            # im = tf.squeeze(sims[idx])
            im = sims[idx]
            sims[idx] = res_fun(
                im,
                model_input_image_size,
                align_corners=True)
        im = tf.squeeze(tf.stack(sims))
        if len(im.get_shape()) < 4:
            im = tf.expand_dims(im, axis=-1)
    else:
        im = res_fun(
            tf.expand_dims(im, axis=0),
            model_input_image_size,
            align_corners=True)
        im = tf.squeeze(im, axis=0)
    return im


def lr_flip_image_label(image, label):
    """Apply a crop to both image and label."""
    image_shape = [int(x) for x in image.get_shape()]
    combined = tf.concat([image, label], axis=-1)
    combined_crop = tf.image.random_flip_left_right(combined)
    image = combined_crop[:, :, :image_shape[-1]]
    label = combined_crop[:, :, image_shape[-1]:]
    return image, label


def ud_flip_image_label(image, label):
    """Apply a crop to both image and label."""
    image_shape = [int(x) for x in image.get_shape()]
    combined = tf.concat([image, label], axis=-1)
    combined_crop = tf.image.random_flip_up_down(combined)
    image = combined_crop[:, :, :image_shape[-1]]
    label = combined_crop[:, :, image_shape[-1]:]
    return image, label


def random_crop_volume(volume, label, target_dims, seed=None):
    """Wrapper for random cropping.

    Modify tf random crop for 4D.
    """
    vol_size = volume.get_shape().as_list()
    tsize = target_dims[:3]
    name = 'combined_crop'

    # Concat volume and label into a single volume for cropping
    combined_volume = tf.concat([volume, label], axis=-1)
    comb_size = combined_volume.get_shape().as_list()
    # crop_size = [comb_size[0]] + tsize + [comb_size[-1]]
    crop_size = tsize + [comb_size[-1]]
    with ops.name_scope(
            name, 'random_crop', [combined_volume, crop_size]) as name:
        combined_volume = ops.convert_to_tensor(combined_volume, name='value')
        crop_size = ops.convert_to_tensor(
            crop_size, dtype=dtypes.int32, name='size')
        vol_shape = array_ops.shape(combined_volume)
        control_flow_ops.Assert(
            math_ops.reduce_all(vol_shape >= crop_size),
            ['Need vol_shape >= vol_size, got ', vol_shape, crop_size],
            summarize=1000)
        limit = vol_shape - crop_size + 1
        offset = tf.random_uniform(
            array_ops.shape(vol_shape),
            dtype=crop_size.dtype,
            maxval=crop_size.dtype.max,
            seed=seed) % limit
        cropped_combined = array_ops.slice(
            combined_volume, offset, crop_size, name=name)
    cropped_volume = cropped_combined[..., :vol_size[-1]]
    cropped_label = cropped_combined[..., vol_size[-1]:]
    return cropped_volume, cropped_label


def center_crop_volume(volume, label, target_dims, seed=None):
    """Wrapper for random cropping.

    Modify tf random crop for 4D.
    """
    vol_size = volume.get_shape().as_list()
    tsize = target_dims[:3]
    name = 'combined_crop'

    # Concat volume and label into a single volume for cropping
    combined_volume = tf.concat([volume, label], axis=-1)
    comb_size = combined_volume.get_shape().as_list()
    # crop_size = [comb_size[0]] + tsize + [comb_size[-1]]
    crop_size = tsize + [comb_size[-1]]
    with ops.name_scope(
            name, 'random_crop', [combined_volume, crop_size]) as name:
        combined_volume = ops.convert_to_tensor(combined_volume, name='value')
        crop_size = ops.convert_to_tensor(
            crop_size, dtype=dtypes.int32, name='size')
        vol_shape = array_ops.shape(combined_volume)
        control_flow_ops.Assert(
            math_ops.reduce_all(vol_shape >= crop_size),
            ['Need vol_shape >= vol_size, got ', vol_shape, crop_size],
            summarize=1000)
        offset = tf.constant((
            np.asarray(vol_size) - np.asarray(target_dims)).astype(np.int32))
        cropped_combined = array_ops.slice(
            combined_volume, offset, crop_size, name=name)
    cropped_volume = cropped_combined[:, :, :, :vol_size[-1]]
    cropped_label = cropped_combined[:, :, :, vol_size[-1]:]
    return cropped_volume, cropped_label


def image_flip(image, direction):
    """Wrapper for image flips."""
    im_size = image.get_shape().as_list()
    if direction == 'left_right':
        flip_function = tf.image.random_flip_left_right
    elif direction == 'up_down':
        flip_function = tf.image.random_flip_up_down
    else:
        raise NotImplementedError

    if len(im_size) == 3:
        return flip_function(image)
    elif len(im_size) == 4:
        if im_size[-1] > 1:
            raise NotImplementedError
        trans_image = tf.transpose(tf.squeeze(image), [1, 2, 0])
        flip_image = tf.expand_dims(
            tf.transpose(flip_function(trans_image), [2, 0, 1]), axis=-1)
        return flip_image
    else:
        raise NotImplementedError


def greyscale_augment(volume, contrast, brightness):
    """Augment volume contrast and brightness."""
    assert contrast is not None, 'Contrast is none.'
    assert brightness is not None, 'Brightness is none.'
    volume = tf.image.random_contrast(image=volume, lower=0., upper=contrast)
    volume = tf.image.random_brightness(image=volume, max_delta=brightness)
    return volume


def flip(volume, label, direction, rtn=False):
    """Flip volume and label pair in a direction."""
    if rtn:
        return volume, label

    # Concat volume and label into a single volume for flipping
    vol_size = volume.get_shape().as_list()
    combined_volume = tf.concat([volume, label], axis=-1)
    if direction == 'lr':
        combined_volume = array_ops.reverse(combined_volume, [1])
    elif direction == 'ud':
        combined_volume = array_ops.reverse(combined_volume, [2])
    elif direction == 'depth':
        combined_volume = array_ops.reverse(combined_volume, [0])
    else:
        raise NotImplementedError('Direction: %s not implemented.')
    flipped_volume = combined_volume[..., :vol_size[-1]]
    flipped_label = combined_volume[..., vol_size[-1]:]
    return flipped_volume, flipped_label


def random_flip(volume, label, direction, threshold=0.5, seed=None):
    """Random flip volume and label pair in a direction."""
    flipped_volume, flipped_label = tf.cond(
        tf.greater(
            tf.random_uniform([], 0, 1.0, seed=seed), threshold),
        lambda: flip(volume, label, direction),
        lambda: flip(volume, label, direction, rtn=True))
    return flipped_volume, flipped_label


def slip(volume, xo, yo, x, y, pivot):
    """Do the slip misalign."""
    # Shift
    new_data = tf.zeros_like(volume)
    xmin = tf.maximum(xo, 0)
    ymin = tf.maximum(yo, 0)
    xmax = tf.minimum(xo, 0) + x
    ymax = tf.minimum(yo, 0) + y
    new_data[:, ymin:ymax, xmin:xmax, :] = volume[
        :, ymin:ymax, xmin:xmax, :]

    # Slip
    xmin = tf.maximum(-xo, 0)
    ymin = tf.maximum(-yo, 0)
    xmax = tf.minimum(-xo, 0) + x
    ymax = tf.minimum(-yo, 0) + y
    new_data[pivot, ymin:ymax, xmin:xmax, :] = volume[
        pivot, ymin:ymax, xmin:xmax, :]
    return new_data


def no_slip(volume, xo, yo, x, y, pivot):
    """Do the no-slip misalign."""
    # First shift
    new_data = tf.zeros_like(volume)
    xmin = tf.maximum(xo, 0)
    ymin = tf.maximum(yo, 0)
    xmax = tf.minimum(xo, 0) + x
    ymax = tf.minimum(yo, 0) + y
    new_data[0:pivot, ymin:ymax, xmin:xmax, :] = volume[
        0:pivot, ymin:ymax, xmin:xmax, :]

    # Shift opposite
    xmin = tf.maximum(-xo, 0)
    ymin = tf.maximum(-yo, 0)
    xmax = tf.minimum(-xo, 0) + x
    ymax = tf.minimum(-yo, 0) + y
    new_data[pivot:, ymin:ymax, xmin:xmax, :] = volume[
        pivot:, ymin:ymax, xmin:xmax, :]
    return new_data


# def misalign(volume, slip_threshold=0.3, seed=None, max_trans=15):
#     """Apply the seung misalign algorithm."""
#     vol_shape = volume.get_shape().as_list()
#     x_offset = tf.round(
#         tf.random_uniform([], -max_trans, max_trans, seed=seed))
#     x_offset = tf.cast(x_offset, tf.int32)
#     y_offset = tf.round(
#         tf.random_uniform([], -max_trans, max_trans, seed=seed))
#     x_offset = tf.cast(x_offset, tf.int32)
#     y_offset = tf.cast(y_offset, tf.int32)
#     pivot = tf.round(
#         tf.random_uniform([], 0, vol_shape[0] - 1, seed=seed))
#     pivot = tf.cast(pivot, tf.int32)
#     slip = tf.greater(
#         tf.random_uniform([], 0, 1.0, seed=seed), slip_threshold)
#     volume = tf.cond(
#         slip,
#         lambda: no_slip(
#             volume=volume,
#             xo=x_offset,
#             yo=y_offset,
#             x=vol_shape[2],
#             y=vol_shape[1],
#             pivot=pivot),
#         lambda: slip(
#             volume=volume,
#             xo=x_offset,
#             yo=y_offset,
#             x=vol_shape[2],
#             y=vol_shape[1],
#             pivot=pivot))
#     return volume


def missing(volume, max_sec):
    """Tensorflow version of Seung's missing section op.

    Insert random slice where rand > max_sec."""
    vol_size = volume.get_shape().as_list()
    z_dim = vol_size[0]
    # num_sec = tf.round(tf.random_uniform([], 1, max_sec))

    # # Fill-out value.
    # updates = tf.random_uniform([max_sec], 0, 1)

    # # Create mask vector
    # mask = tf.ones([z_dim, 1, 1])

    # # Mask slices
    # indices = tf.gather(
    #     tf.random_shuffle(tf.range(z_dim)),
    #     tf.cast(tf.range(num_sec), tf.int32))
    # mask = tf.scatter_sub(mask, indices, updates)
    rand_mask = tf.random_uniform([z_dim, 1, 1, 1], 0, 1)
    mask = tf.cast(
        tf.greater_equal(
            rand_mask, max_sec),
        tf.float32)
    flip = tf.abs(mask - 1)
    rand_vec = (flip * rand_mask)
    volume = (volume * mask) + rand_vec
    return volume


def misalign(volume, max_trans, interpolation='NEAREST'):
    """Reimplementation of Seung misalign function."""
    vol_size = volume.get_shape().as_list()
    pivot = tf.random_uniform([], 1, vol_size[0] - 1)
    fh = tf.gather(volume, tf.cast(tf.range(pivot), tf.int32))
    sh = tf.gather(volume, tf.cast(tf.range(pivot + 1, vol_size[0]), tf.int32))
    # base = tf.convert_to_tensor(
    #     np.tile([1, 0, 0, 0, 1, 0, 0, 0], [vol_size[0], 1]),
    #     dtype=tf.float32)
    # mask = tf.convert_to_tensor(
    #     np.tile([0, 0, 1, 0, 0, 1, 0, 0], [vol_size[0], 1]),
    #     dtype=tf.float32)
    # random_shift = tf.random_uniform(
    #     [1, 8],
    #     minval=-max_trans,
    #     maxval=max_trans,
    #     dtype=tf.float32)
    # transforms = base + random_shift * mask
    # fh = tf.contrib.image.transform(
    #     fh,
    #     transforms)
    # transforms = base - random_shift * mask
    # sh = tf.contrib.image.transform(
    #     sh,
    #     transforms)
    txy = tf.round(tf.random_uniform([4], minval=-max_trans, maxval=max_trans))
    fh_transforms = [1, 0, txy[0], 0, 1, txy[1], 0, 0]
    fh = tf.contrib.image.transform(
        fh,
        fh_transforms,
        interpolation)
    sh_transforms = [1, 0, txy[2], 0, 1, txy[3], 0, 0]
    sh = tf.contrib.image.transform(
        sh,
        sh_transforms,
        interpolation)
    volume = tf.concat([fh, sh], axis=0)
    volume.set_shape(vol_size)
    return volume


def gauss_kernel(kernlen=7, nsig=3, channels=1):
    """Create gaussian kernel"""
    interval = (2 * nsig + 1.) / kernlen
    x = np.linspace(
        -nsig - interval / 2.,
        nsig + interval / 2.,
        kernlen + 1)
    kern1d = np.diff(stats.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return out_filter


def renorm(volume, minv, maxv):
    """Normalize volume to minv, maxv."""
    # cmin = tf.reduce_min(volume)
    # cmax = tf.reduce_max(volume)
    # nv = (volume - cmin) / (cmax - cmin)

    # Scale to minv, maxv
    return (volume * (maxv - minv)) + minv


def smooth(volume):
    """Smooth the volume with a gabor."""
    kernel = tf.constant(gauss_kernel(kernlen=28))
    ch_1 = volume[..., 0]
    ch_2 = volume[..., 1]
    smooth_ch_1 = tf.squeeze(tf.nn.conv3d(input=tf.expand_dims(tf.expand_dims(volume[..., 0], 0), -1),filter=tf.expand_dims(kernel, 0), strides=[1, 1, 1, 1, 1], padding='SAME'), 0)
    smooth_ch_2 = tf.squeeze(tf.nn.conv3d(input=tf.expand_dims(tf.expand_dims(volume[..., 1], 0), -1),filter=tf.expand_dims(kernel, 0), strides=[1, 1, 1, 1, 1], padding='SAME'), 0)
    smooth_vol = tf.concat((smooth_ch_1, smooth_ch_2), -1)
    smooth_vol = smooth_vol / tf.reduce_max(smooth_vol)
    return smooth_vol


def blur(volume, max_rad, min_rad, max_sec):
    """Apply a blur to a random location in the volume."""
    vol_size = volume.get_shape().as_list()
    z_dim = vol_size[0]
    x, y = tf.meshgrid(tf.range(vol_size[1]), tf.range(vol_size[2]))
    grid = tf.cast(tf.transpose(tf.stack([x, y]), (1, 2, 0)), tf.float32)
    coord = tf.round(tf.random_uniform([1, 1, 2], 0, vol_size[1]))
    # grid = tf.sigmoid(tf.reduce_sum((grid - coord) ** 2, axis=-1))
    grid = tf.reduce_sum((grid - coord) ** 2, axis=-1)
    grid /= tf.reduce_max(grid)
    rand_thresh = tf.random_uniform([], min_rad, max_rad)
    grid = tf.expand_dims(
        tf.expand_dims(
            tf.cast(
                tf.less_equal(grid, rand_thresh),
                tf.float32),
            axis=-1),
        axis=-1)
    kernel = tf.constant(gauss_kernel())
    smooth_grid = tf.squeeze(
        tf.nn.conv2d(
            input=grid,
            filter=kernel,
            strides=[1, 1, 1, 1],
            padding='SAME'))
    smooth_volume = tf.squeeze(
        tf.nn.conv2d(  # use depthwise for variable channels
            input=volume,
            filter=kernel,
            strides=[1, 1, 1, 1],
            padding='SAME'))
    smooth_grid = tf.expand_dims(
        tf.expand_dims(
            smooth_grid,
            axis=0),
        axis=-1)
    smooth_volume = tf.expand_dims(
        tf.expand_dims(
            smooth_volume,
            axis=0),
        axis=-1)
    minv = tf.reduce_min(volume)
    maxv = tf.reduce_max(volume)
    smooth_volume = renorm(volume, minv, maxv)
    smooth_grid = renorm(smooth_grid, 0, 1)
    pos = smooth_grid * smooth_volume
    neg = (tf.abs(1 - smooth_grid)) * volume
    blur_volume = pos + neg

    # Apply blur to max_sec slices
    rand_mask = tf.random_uniform([z_dim, 1, 1, 1], 0, 1)
    mask = tf.cast(
        tf.greater_equal(
            rand_mask, max_sec),
        tf.float32)
    flip = tf.abs(mask - 1)
    blur_volume *= flip
    volume = (volume * mask) + (blur_volume * flip)
    return pos + neg


def augment(
        volume,
        label,
        augmentations,
        input_shape,
        label_shape):
    """Coordinating image augmentations for both image and heatmap."""
    if not isinstance(augmentations, list):
        augmentations = [augmentations]
    for augmentation in augmentations:
        aug, params = list(augmentation.items())[0]
        if 'normalize_volume' in aug:
            volume = params(volume)
        elif 'normalize_label' in aug:
            label = params(label)
        elif 'assert_max_volume' in aug:
            tf.assert_less_equal(
                tf.reduce_max(volume),
                params,
                message='Failed volume value assert.')
        elif 'assert_max_label' in aug:
            tf.assert_less_equal(
                tf.reduce_max(label),
                params,
                message='Failed label value assert.')
        elif 'random_crop' in aug:
            if len(params):
                input_shape = params
            volume, label = random_crop_volume(
                volume=volume,
                label=label,
                target_dims=input_shape)
        elif 'center_crop' in aug:
            if len(params):
                input_shape = params
            volume, label = center_crop_volume(
                volume=volume,
                label=label,
                target_dims=input_shape)
        elif 'greyscale' in aug or 'pixel' in aug:  # TF
            if 'contrast' in list(params.keys()):
                contrast = params['contrast']
            else:
                contrast = 0.3
            if 'brightness' in list(params.keys()):
                brightness = params['brightness']
            else:
                brightness = 0.3
            volume = greyscale_augment(
                volume=volume,
                contrast=contrast,
                brightness=brightness)
        elif 'clip' in aug:
            if 'high' in list(params.keys()):
                high = params['high']
            else:
                high = 1
            if 'low' in list(params.keys()):
                low = params['low']
            else:
                low = 0
            volume = tf.maximum(
                tf.minimum(volume, high), low)
        elif 'lr_flip' in aug or 'flip_lr' in aug:
            volume, label = random_flip(
                volume=volume,
                label=label,
                direction='lr')
        elif 'ud_flip' in aug or 'flip_ud' in aug:
            volume, label = random_flip(
                volume=volume,
                label=label,
                direction='ud')
        elif 'depth_flip' in aug or 'flip_depth' in aug:
            volume, label = random_flip(
                volume=volume,
                label=label,
                direction='depth')
        elif 'smooth' in aug:
            label = smooth(label)
        elif 'blur' in aug:
            if 'max_rad' in list(params.keys()):
                max_rad = params['max_rad']
            else:
                max_rad = 0.04
            if 'min_rad' in list(params.keys()):
                min_rad = params['min_rad']
            else:
                min_rad = 0.004
            if 'max_sec' in list(params.keys()):
                max_sec = params['max_sec']
            else:
                max_sec = 0.15
            volume = blur(
                volume,
                max_rad=max_rad,
                min_rad=min_rad,
                max_sec=max_sec)
        elif 'missing' in augmentation:
            if 'max_sec' in list(params.keys()):
                max_sec = params['max_sec']
            else:
                max_sec = 0.15
            volume = missing(volume, max_sec=max_sec)
        elif 'misalign' in augmentation:
            if 'max_trans' in list(params.keys()):
                max_trans = params['max_trans']
            else:
                max_trans = 15
            volume = misalign(volume, max_trans=max_trans)
        elif 'warp' in aug:
            if 'theta' in list(params.keys()):
                theta = params['theta']
            else:
                theta = 22.
            vol_size = volume.get_shape().as_list()
            angle_rad = (theta / 180.) * math.pi
            angles = tf.random_uniform([], -angle_rad, angle_rad)
            transform = tf.contrib.image.angles_to_projective_transforms(
                angles,
                vol_size[1],
                vol_size[2])
            volume = tf.contrib.image.transform(
                volume,
                tf.contrib.image.compose_transforms(transform),
                interpolation='BILINEAR')
        else:
            raise NotImplementedError('Augmentation: %s' % aug)
    return volume, label


def decode_data(features, reader_settings):
    """Decode data from TFrecords."""
    if features.dtype == tf.string:
        return tf.decode_raw(
            features,
            reader_settings)
    else:
        return tf.cast(
            features,
            reader_settings)


def read_and_decode(
        filename_queue,
        input_shape,
        tf_dict,
        tf_reader_settings,
        augmentations,
        number_of_files,
        label_shape):
    """Read and decode tensors from tf_records and apply augmentations."""
    reader = tf.TFRecordReader()

    # Switch between single/multi-file reading
    if number_of_files == 1:
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features=tf_dict)
    else:
        _, serialized_examples = reader.read_up_to(
            filename_queue,
            num_records=number_of_files)
        features = tf.parse_example(
            serialized_examples,
            features=tf_dict)

    # Handle decoding of each element
    volume = decode_data(
        features=features['volume'],
        reader_settings=tf_reader_settings['volume']['dtype'])
    label = decode_data(
        features=features['label'],
        reader_settings=tf_reader_settings['label']['dtype'])

    # Reshape each element
    volume = tf.reshape(volume, tf_reader_settings['volume']['reshape'])
    if tf_reader_settings['label']['reshape'] is not None:
        label = tf.reshape(label, tf_reader_settings['label']['reshape'])
    volume = tf.cast(volume, tf.float32)
    label = tf.cast(label, tf.float32)

    # 3D image augmentations.
    volume, label = augment(
        volume=volume,
        label=label,
        augmentations=augmentations,
        input_shape=input_shape,
        label_shape=label_shape)
    return volume, label


def inputs_mt(
        dataset,
        batch_size,
        input_shape,
        label_shape,
        tf_dict,
        data_augmentations,
        num_epochs,
        tf_reader_settings,
        shuffle,
        number_of_files=1,
        min_after_dequeue=500,
        capacity=5000,
        num_threads=1):
    """Read tfrecords and prepare them for queueing."""
    capacity *= batch_size  # min_after_dequeue + 5 * batch_size

    # Start data loader loop
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [dataset], num_epochs=num_epochs)
        # Even when reading in multiple threads, share the filename
        # queue.
        volume_data, label_data = [], []
        for idx in range(batch_size):
            volumes, labels = read_and_decode(
                filename_queue=filename_queue,
                input_shape=input_shape,
                label_shape=label_shape,
                tf_dict=tf_dict,
                tf_reader_settings=tf_reader_settings,
                augmentations=data_augmentations,
                number_of_files=number_of_files)
            volume_data += [tf.expand_dims(volumes, axis=0)]
            label_data += [tf.expand_dims(labels, axis=0)]

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        batch_data = [
            tf.concat(volume_data, axis=0),  # CHANGE TO STACK
            tf.concat(label_data, axis=0)
        ]
        if shuffle:
            volumes, labels = tf.train.shuffle_batch(
                batch_data,
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=capacity,
                enqueue_many=True,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=min_after_dequeue)
        else:
            volumes, labels = tf.train.batch(
                batch_data,
                batch_size=batch_size,
                num_threads=num_threads,
                enqueue_many=True,
                capacity=capacity)
        return volumes, labels


def inputs(
        dataset,
        batch_size,
        input_shape,
        label_shape,
        tf_dict,
        data_augmentations,
        num_epochs,
        tf_reader_settings,
        shuffle,
        number_of_files=1,
        min_after_dequeue=10,
        capacity=25,
        num_threads=1):
    """Read tfrecords and prepare them for queueing."""
    capacity *= batch_size  # min_after_dequeue + 5 * batch_size

    # Start data loader loop
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [dataset],
            num_epochs=num_epochs)
        # Even when reading in multiple threads, share the filename
        # queue.
        batch_data = read_and_decode(
            filename_queue=filename_queue,
            input_shape=input_shape,
            label_shape=label_shape,
            tf_dict=tf_dict,
            tf_reader_settings=tf_reader_settings,
            augmentations=data_augmentations,
            number_of_files=number_of_files)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        if shuffle:
            volumes, labels = tf.train.shuffle_batch(
                batch_data,
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=capacity,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=min_after_dequeue)
        else:
            volumes, labels = tf.train.batch(
                batch_data,
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=capacity)
        return volumes, labels


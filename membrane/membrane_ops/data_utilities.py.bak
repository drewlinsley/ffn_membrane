from __future__ import division
import warnings
import numpy as np
# import cv2
from skimage import transform
from skimage.filters import scharr
# from ops.augmentations import blur, misalign, missing, warp, pixel


def warp2d(volume, label, angle=30, resize_rate=0.9):
    """Apply warp to 2d images."""
    vol_shape = volume.shape
    label_shape = label.shape
    sh = np.random.random() / 4 - 0.25
    # rx = np.sign(np.random.random() - 0.5)
    # ry = np.sign(np.random.random() - 0.5)
    # scx = np.random.random() / 8 - 0.25 + (rx * 1)
    # scy = np.random.random() / 8 - 0.25 + (ry * 1)
    rotate_angle = np.random.random() / 180 * np.pi * angle
    afine_tf = transform.AffineTransform(
        shear=sh,
        # scale=[scx, scy],
        rotation=rotate_angle)
    volume = transform.warp(
        volume.squeeze(),
        inverse_map=afine_tf,
        mode='constant').reshape(vol_shape)
    label = transform.warp(
        label.squeeze(),
        inverse_map=afine_tf,
        mode='constant').reshape(label_shape)
    return volume, label


def elastic_transform(image, alpha, sigma):
    """ Elastic deformation of input image
    Inputs: image - input image
            alpha - scaling factor
            sigma - standard deviation of gaussian filter
    Outputs: distorted_image - image that has elastic deformation
    """
    shape = image.shape
    dx = gaussian_filter((
        np.random.rand(shape[0], shape[1], shape[2]) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((
        np.random.rand(shape[0], shape[1], shape[2]) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)
    x, y, z = np.meshgrid(
        np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(
        y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    distorted_image = map_coordinates(
        image, indices, order=1, mode='reflect')
    return distorted_image.reshape(shape)


def check_volume(data):
    """Ensure that data is numpy 3D array."""
    assert isinstance(data, np.ndarray)

    if data.ndim == 2:
        data = data[np.newaxis, ...]
    elif data.ndim == 3:
        pass
    elif data.ndim == 4:
        assert data.shape[0] == 1
        data = np.reshape(data, data.shape[-3:])
    else:
        raise RuntimeError('data must be a numpy 3D array')

    assert data.ndim == 3
    return data


def affinitize(img, dst, dtype='float32'):
    """
    Transform segmentation to 3D affinity graph.
    Args:
        img: 3D indexed image, with each index corresponding to each segment.
    Returns:
        ret: affinity graph
    """
    img = check_volume(img)
    ret = np.zeros((1,) + img.shape, dtype=dtype)

    (dz, dy, dx) = dst
    if dz != 0:
        # z-affinity.
        assert dz and abs(dz) < img.shape[0]
        if dz > 0:
            ret[0, dz:, :, :] = (
                img[dz:, :, :] == img[:-dz, :, :]) & (img[dz:, :, :] > 0)
        else:
            dz = abs(dz)
            ret[0, :-dz, :, :] = (
                img[dz:, :, :] == img[:-dz, :, :]) & (img[dz:, :, :] > 0)

    if dy != 0:
        # y-affinity.
        assert dy and abs(dy) < img.shape[2]
        if dy > 0:
            ret[0, :, dy:, :] = (
                img[:, dy:, :] == img[:, :-dy, :]) & (img[:, dy:, :] > 0)
        else:
            dy = abs(dy)
            ret[0, :, :-dy, :] = (
                img[:, dy:, :] == img[:, :-dy, :]) & (img[:, dy:, :] > 0)

    if dx != 0:
        # x-affinity.
        assert dx and abs(dx) < img.shape[1]
        if dx > 0:
            ret[0, :, :, dx:] = (
                img[:, :, dx:] == img[:, :, :-dx]) & (img[:, :, dx:] > 0)
        else:
            dx = abs(dx)
            ret[0, :, :, :-dx] = (
                img[:, :, dx:] == img[:, :, :-dx]) & (img[:, :, dx:] > 0)
    return np.squeeze(ret)


def derive_affinities(affinity, label_volume, long_range=True):
    """Derive affinities from label_volume."""
    if affinity:
        warnings.warn('Affinity argument', DeprecationWarning)
    # distances = np.rot90(np.eye(affinity)).astype(int)
    # Hard code this to be long-range
    distances = np.rot90(np.eye(3)).astype(int)
    if long_range:
        lr = [
            [0, 0, 3],
            [0, 3, 0],
            [2, 0, 0],
            [0, 0, 9],
            [0, 9, 0],
            [3, 0, 0],
            [0, 0, 27],
            [0, 27, 0],
            [4, 0, 0],
        ]
        distances = np.concatenate((distances, lr), axis=0)
    ground_truth_affinities = []
    for i in range(len(distances)):
        aff = affinitize(label_volume, dst=distances[i])
        ground_truth_affinities += [aff.astype(int)]
    label_volume = np.array(
        ground_truth_affinities).transpose(1, 2, 3, 0)
    return label_volume


def create_boundary_map(np_masks):
    adjusted_masks = []
    for i in range(np.shape(np_masks)[0]):
        mask = np.squeeze(np_masks[i, :, :])
        adjusted_mask = scharr(mask)
        adjusted_mask[adjusted_mask > 0] = 1
        adjusted_mask = adjusted_mask.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        adjusted_mask = cv2.dilate(
            adjusted_mask,
            kernel,
            iterations=1)
        adjusted_mask = np.invert(adjusted_mask)  # * 255
        adjusted_masks.append(adjusted_mask)

    adjusted_masks = np.array(adjusted_masks)
    return adjusted_masks - 254


def apply_flip(x, direction):
    """Apply a flip to a volume."""
    if direction == 'lr':
        return np.fliplr(x)
    elif direction == 'ud':
        return np.flipud(x)
    else:
        raise NotImplementedError('Direction: %s is not valid.' % direction)
    return x


def random_crop_volume(volume, target_dims):
    """Derive random indices for cropping volumes."""
    vshape = volume.shape
    if len(vshape) == 5:
        # Cropping on the 2nd and 3rd dims
        h_diff = vshape[2] - target_dims[1]
        w_diff = vshape[3] - target_dims[2]
        if h_diff == 0 or w_diff == 0:
            return volume, None
        else:
            h_crop = np.random.randint(low=0, high=vshape[2] - target_dims[1])
            w_crop = np.random.randint(low=0, high=vshape[3] - target_dims[2])
            return volume[
                :,
                :,
                h_crop:h_crop + target_dims[1],
                w_crop:w_crop + target_dims[2],
                :], (h_crop, w_crop)
    elif len(vshape) == 4:
        # Cropping on the 2nd and 3rd dims
        h_crop = np.random.randint(low=0, high=vshape[1] - target_dims[1])
        w_crop = np.random.randint(low=0, high=vshape[2] - target_dims[2])
        if h_crop == 0 and w_crop == 0:
            return volume, None
        else:
            return volume[
                :,
                h_crop:h_crop + target_dims[1],
                w_crop:w_crop + target_dims[2],
                :], (h_crop, w_crop)
    else:
        raise NotImplementedError('Volume must be 4D or 5D.')


def center_crop_volume(volume, target_dims):
    """Derive center indices for cropping volumes."""
    vshape = volume.shape
    assert len(vshape) == 5, 'Volume must be 5D.'
    if len(vshape) == 5:
        # Cropping on the 2nd and 3rd dims
        h_crop = vshape[2] - target_dims[1]
        w_crop = vshape[3] - target_dims[2]
        if h_crop == 0 and w_crop == 0:
            return volume, None
        else:
            return volume[
                :,
                :,
                h_crop:h_crop + target_dims[1],
                w_crop:w_crop + target_dims[2],
                :], (h_crop, w_crop)
    elif len(vshape) == 4:
        h_crop = vshape[1] - target_dims[1]
        w_crop = vshape[2] - target_dims[2]
        if h_crop == 0 and w_crop == 0:
            return volume, None
        else:
            return volume[
                :,
                h_crop:h_crop + target_dims[1],
                w_crop:w_crop + target_dims[2],
                :], (h_crop, w_crop)
    else:
        raise NotImplementedError('Volume must be 4D or 5D.')


def crop_volume(volume, target_dims, indices):
    """Crop a volume with random indices."""
    vshape = volume.shape
    if len(vshape) == 5:
        return volume[
            :,
            :,
            indices[0]:indices[0] + target_dims[1],
            indices[1]:indices[1] + target_dims[2],
            :]
    elif len(vshape) == 4:
        return volume[
            :,
            indices[0]:indices[0] + target_dims[1],
            indices[1]:indices[1] + target_dims[2],
            :]
    else:
        raise NotImplementedError('Volume must be 4D or 5D.')


def apply_augmentations(
        volume,
        label,
        augmentations,
        input_shape,
        label_shape):
    """Loop through augmentation list of dicts."""
    for aug in augmentations:
        augmentation, params = aug.items()[0]
        volume, label = interpret_augmentation(
            augmentation=augmentation,
            params=params,
            volume=volume,
            label=label,
            input_shape=input_shape,
            label_shape=label_shape)
    return volume, label


def update_defaults(defaults, params):
    """Update defaults with params if needed."""
    if isinstance(params, dict):
        for k, v in params.iteritems():
            if k in defaults.keys():
                defaults[k] = v
    return defaults


def interpret_augmentation(
        volume,
        label,
        augmentation,
        params,
        input_shape,
        label_shape):
    """Interpret and apply augmentations to volume and label."""
    if 'normalize_volume' in augmentation:
        volume = params(volume)
    elif 'normalize_label' in augmentation:
        label = params(label)
    elif 'assert_max_volume' in augmentation:
        assert np.max(volume) <= params, 'Failed volume value assert.'
    elif 'assert_max_label' in augmentation:
        assert np.max(label) <= params, 'Failed label value assert.'
    elif 'random_crop' in augmentation:  # TF
        if len(params):
            input_shape = params
        volume, crop_idx = random_crop_volume(
            volume=volume,
            target_dims=input_shape)
        if crop_idx is not None:
            label = crop_volume(
                volume=label,
                target_dims=input_shape,
                indices=crop_idx)
    elif 'center_crop' in augmentation:  # TF
        if len(params):
            input_shape = params
        volume, crop_idx = center_crop_volume(
            volume=volume,
            target_dims=input_shape)
        if crop_idx is not None:
            label = crop_volume(
                volume=label,
                target_dims=input_shape,
                indices=crop_idx)
    elif 'blur' in augmentation:  # TF
        blur_defaults = {
            'max_sec': 1,
            'skip_ratio': 0.3,
            'mode': 'full',
            'sigma_max': 5.}
        blur_defaults = update_defaults(blur_defaults, params)
        blur_augment = blur.BlurAugment(**blur_defaults)
        volume = blur_augment.augment(volume)
    elif 'misalign' in augmentation:
        misalign_defaults = {
            'max_trans': 15,
            'slip_ratio': 0.3,
            'skip_ratio': 0.}
        misalign_defaults = update_defaults(misalign_defaults, params)
        misalign_augment = misalign.MisalignAugment(**misalign_defaults)
        spec = dict()
        spec['volume'] = tuple(input_shape[:3])
        spec['label'] = tuple(label_shape[:3])
        misalign_augment.prepare(spec)
        for idx, (vol, lab) in enumerate(zip(volume, label)):
            sample = misalign_augment.augment(
                sample={'volume': vol, 'label': lab})
            volume[idx] = sample['volume']
            label[idx] = sample['label']
    elif 'warp' in augmentation:
        vol_shape = volume.shape
        if vol_shape[1] == 1:
            for idx, (vol, lab) in enumerate(zip(volume, label)):
                it_volume, it_label = warp2d(
                    volume=volume[idx],
                    label=label[idx])
                volume[idx] = it_volume
                label[idx] = it_label
        else:
            # 3D case
            warp_defaults = {
                'skip_ratio': 0.3}
            warp_defaults = update_defaults(warp_defaults, params)
            spec = dict()
            spec['volume'] = tuple(vol_shape[1:4])
            spec['label'] = tuple(label.shape[1:4])
            warp_augment = warp.WarpAugment(**warp_defaults)
            warp_augment.prepare(spec, imgs=['volume', 'label'])
            for idx, (vol, lab) in enumerate(zip(volume, label)):
                sample = warp_augment.augment(
                    sample={'volume': vol, 'label': lab},
                    imgs=['volume', 'label'])
                volume[idx] = sample['volume']
                label[idx] = sample['label']
    elif 'missing' in augmentation:
        missing_defaults = {
            'max_sec': 1,
            'mode': 'full',
            'consecutive': False,
            'random_color': False,
            'skip_ratio': 0.3}
        missing_defaults = update_defaults(missing_defaults, params)
        missing_augment = missing.MissingAugment(**missing_defaults)
        spec = dict()
        spec['volume'] = tuple(input_shape[:3])
        spec['label'] = tuple(label_shape[:3])
        missing_augment.prepare(spec)
        for idx, (vol, lab) in enumerate(zip(volume, label)):
            sample = missing_augment.augment(
                sample={'volume': vol, 'label': lab},
                imgs=['volume', 'label'])
            volume[idx] = sample['volume']
            label[idx] = sample['label']
    elif 'lr_flip' in augmentation or 'flip_lr' in augmentation:  # TF
        flip_prop = 0.5
        if len(params):
            flip_prop = params
        assert flip_prop > 0. and flip_prop < 1.
        flip = np.random.rand() > flip_prop
        if flip:
            volume = apply_flip(volume, 'lr')
            label = apply_flip(label, 'lr')
    elif 'up_flip' in augmentation or 'flip_ud' in augmentation:  # TF
        flip_prop = 0.5
        if len(params):
            flip_prop = params
        assert flip_prop > 0. and flip_prop < 1.
        flip = np.random.rand() > flip_prop
        if flip:
            volume = apply_flip(volume, 'ud')
            label = apply_flip(label, 'ud')
    elif 'uniform' in augmentation:
        if len(params):
            w = params[0]
        else:
            w = 0.
        noise = np.random.uniform(low=-w, high=w, size=input_shape)
        volume = np.minimum(1, np.maximum(0, volume + noise))
    elif 'pixel' in augmentation:  # TF
        pixel_defaults = {
            'mode': '3D',
            'skip_ratio': 0.3,
            'CONTRAST_FACTOR': 0.3,
            'BRIGHTNESS_FACTOR': 0.3
        }
        pixel_defaults = update_defaults(pixel_defaults, params)
        pixel_augment = pixel.GreyAugment(**pixel_defaults)
        spec = dict()
        spec['volume'] = tuple(input_shape[:3])
        pixel_augment.prepare(spec)
        sample = pixel_augment.augment(
            sample={'volume': volume},
            imgs=['volume'])
        volume = sample['volume']
    elif 'flip_label' in augmentation:
        label = np.abs(1. - label)
    return volume, label


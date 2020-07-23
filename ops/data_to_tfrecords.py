"""Routines for encoding data into TFrecords."""
import os
import numpy as np
import tensorflow as tf
from scipy import misc
from tqdm import tqdm
from utils import image_processing
from utils.data_utilities import apply_augmentations, derive_affinities


def bytes_feature(values):
    """Bytes features for writing TFRecords."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


TARGETS = {
    'volume': bytes_feature,
    'label': bytes_feature
}
ND_TRAIN_AUGMENTATIONS = [
    {'min_max_native_normalization': []},
    # {'normalize_volume': lambda x: x / 255.},
    # {'warp': {}},
    # {'random_crop': []},
    {'pixel': {}},
    {'misalign': {}},
    {'blur': {}},
    {'missing': {}},
    {'flip_lr': []},
    {'flip_ud': []},
]
ND_TEST_AUGMENTATIONS = [
    {'min_max_native_normalization': []},
]
INPUT_SHAPE = (18, 384, 384, 1)
LABEL_SHAPE = (18, 384, 384, 12)


def convert_norm(meta):
    """Create the native normalization."""
    return {'normalize_volume': lambda x: (
        x - meta['min_val']) / (
            meta['max_val'] - meta['min_val'])}


def convert_z(meta):
    """Create the native normalization."""
    return {'normalize_volume': lambda x: (
        x - meta['mean']) / meta['std']}


def uint8_normalization():
    """Return a x/255. normalization clipped to [0, 1]."""
    return {'normalize_volume': lambda x: tf.minimum(x / 255., 1)}


def check_augmentations(augmentations, meta={'min_val':0., 'max_val': 255.}):
    """Adjust augmentations based on keywords."""
    if isinstance(augmentations, list):
        for idx, augmentation in enumerate(augmentations):
            if 'min_max_native_normalization' in list(augmentation.keys()):
                augmentations[idx] = convert_norm(meta)
            elif 'native_normalization_z' in list(augmentation.keys()):
                augmentations[idx] = convert_z(meta)
            elif 'uint8_normalization' in list(augmentation.keys()):
                augmentations[idx] = uint8_normalization()
    else:
        print('WARNING: No normalization requested.')
    return augmentations


def load_image(f, im_size, repeat_image=False):
    """Load image and convert it to a 4D tensor."""
    image = misc.imread(f).astype(np.float32)
    if len(image.shape) < 3 and repeat_image:  # Force H/W/C
        image = np.repeat(image[:, :, None], im_size[-1], axis=-1)
    return image


def normalize(im):
    """Normalize to [0, 1]."""
    min_im = im.min()
    max_im = im.max()
    return (im - min_im) / (max_im - min_im)


def create_example(data_dict):
    """Create entry in tfrecords."""
    data_dict = {k: v for k, v in data_dict.items() if v is not None}
    return tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
            # Features has a map of string to Feature proto objects
            feature=data_dict
        )
    )


def preprocess_image(image, preprocess, im_size):
    """Preprocess image files before encoding in TFrecords."""
    if 'crop_center' in preprocess:
        image = image_processing.crop_center(image, im_size)
    elif 'crop_center_resize' in preprocess:
        im_shape = image.shape
        min_shape = np.min(im_shape[:2])
        crop_im_size = [min_shape, min_shape, im_shape[-1]]
        image = image_processing.crop_center(image, crop_im_size)
        image = image_processing.resize(image, im_size)
    elif 'resize' in preprocess:
        image = image_processing.resize(image, im_size)
    elif 'pad_resize' in preprocess:
        image = image_processing.pad_square(image)
        image = image_processing.resize(image, im_size)
    return image.astype(np.float32)


def encode_tf(encoder, x):
    """Process data for TFRecords."""
    encoder_name = encoder.__name__
    if 'bytes' in encoder_name:
        return encoder(x.tostring())
    else:
        return encoder(x)


def prep_record(
        it_f,
        it_l,
        im_size,
        label_size,
        targets,
        repeat_image,
        preprocess,
        normalize_im,
        store_z,
        means):
    """Perform operations to prepare a record."""
    if isinstance(it_f, str):
        if '.npy' in it_f:
            volume = np.load(it_f)
        else:
            volume = load_image(
                it_f,
                im_size,
                repeat_image=repeat_image)
        if len(volume.shape) > 1:
            volume = preprocess_image(volume, preprocess, im_size)
    else:
        volume = preprocess_image(it_f, preprocess, im_size)
    if normalize_im:
        volume = normalize(volume)
    volume = volume.astype(np.float32)
    if store_z:
        means += [volume]
    else:
        means += volume
    if isinstance(it_l, str):
        if '.npy' in it_l:
            label = np.load(it_l)
        else:
            label = load_image(
                it_l,
                label_size,
                repeat_image=False).astype(np.float32)
        if len(label.shape) > 1:
            label = preprocess_image(label, preprocess, label_size)
    else:
        label = it_l
        if isinstance(label, np.ndarray) and len(label.shape) > 1:
            label = preprocess_image(
                label, preprocess, label_size)
    label = label.astype(np.float32)
    data_dict = {
        'volume': encode_tf(targets['volume'], volume),
        'label': encode_tf(targets['label'], label)
    }
    example = create_example(data_dict)
    return example, means


def check_example(example, tfrecord_writer, image_count):
    """Check and write tfrecords."""
    if example is not None:
        # Keep track of how many images we use
        image_count += 1
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        tfrecord_writer.write(serialized)
        example = None
    return example, image_count


def process_example(
        it_f,
        it_l,
        means,
        im_size,
        label_size,
        targets,
        repeat_image,
        preprocess,
        normalize_im,
        store_z,
        tfrecord_writer,
        augment,
        augmentations,
        input_shape,
        label_shape,
        image_count,
        metas=None):
    """Process example and return encoding."""
    example_list = []
    if augment:
        # Sample overlapping volumes with input_shape slices
        max_slice = it_f.shape[0] - input_shape[0]
        exp_it_f = np.expand_dims(it_f, axis=0).astype(np.float32)
        if len(exp_it_f.shape) < 5:
            exp_it_f = np.expand_dims(exp_it_f, axis=-1)
        exp_it_l = np.expand_dims(it_l, axis=0).astype(np.float32)
        if len(augmentations) == 1 and 'normalize_volume' in augmentations[0]:
            augment = 1
        for _ in tqdm(list(range(augment)), desc='augmentation', total=augment):
            # Augment it_f and it_l
            if max_slice > 0:
                start_slice = np.random.permutation(max_slice)[0]
            else:
                start_slice = 0
            aug_it_f = exp_it_f[
                :,
                start_slice: start_slice + input_shape[0],
                :,
                :,
                :]
            aug_it_l = exp_it_l[
                :,
                start_slice: start_slice + input_shape[0],
                :,
                :,
                :]
            aug_it_f, aug_it_l = apply_augmentations(
                volume=aug_it_f,
                label=aug_it_l,
                augmentations=augmentations,
                input_shape=im_size,
                label_shape=label_size)
            aug_it_f = np.squeeze(aug_it_f, axis=0)
            aug_it_l = np.squeeze(aug_it_l, axis=0)

            # Write record
            example, means = prep_record(
                it_f=aug_it_f,
                it_l=aug_it_l,
                im_size=im_size,
                label_size=label_size,
                targets=targets,
                repeat_image=repeat_image,
                preprocess=preprocess,
                normalize_im=normalize_im,
                store_z=store_z,
                means=means)
            example, image_count = check_example(
                example=example,
                tfrecord_writer=tfrecord_writer,
                image_count=image_count)
            example_list += [example]
    else:
        example, means = prep_record(
            it_f=it_f,
            it_l=it_l,
            im_size=im_size,
            label_size=label_size,
            targets=targets,
            repeat_image=repeat_image,
            preprocess=preprocess,
            normalize_im=normalize_im,
            store_z=store_z,
            means=means)
        example_list, image_count = check_example(
            example=example,
            tfrecord_writer=tfrecord_writer,
            image_count=image_count)
    return example_list, image_count, means


def create_multidataset(
        files,
        targets,
        ds_name,
        im_size,
        label_size,
        preprocess,
        store_z,
        normalize_im,
        it_ds_name,
        affinity,
        augment,
        held_out,
        repeat_image,
        metas=None,
        add_train_augmentations=[{'random_crop': []}],
        add_test_augmentations=[{'center_crop': []}]):
    """Sequentially load then build datasets."""
    f_string = '_'.join(
        [f.split(os.path.sep)[-1].split('.')[0] for f in files])

    # Train set
    if it_ds_name is None:
        folder = os.path.sep.join(ds_name.split(os.path.sep)[:-1])
        it_ds_name = os.path.join(
            folder,
            '%s_%s.tfrecords' % ('train', f_string))
    train_files = files[:-held_out]
    test_files = files[-held_out:]
    if not isinstance(test_files, list):
        test_files = [test_files]
    with tf.python_io.TFRecordWriter(it_ds_name) as tfrecord_writer:
        image_count = 0
        for f_idx, f in tqdm(
                enumerate(train_files),
                desc='Multidataset train',
                total=len(train_files)):
            dataset = np.load(f)
            volume, label = dataset['volume'], dataset['label']
            if affinity > 1:
                label = derive_affinities(
                    affinity=affinity,
                    label_volume=label)
            it_augmentations = check_augmentations(
                ND_TRAIN_AUGMENTATIONS)
            it_augmentations += add_train_augmentations
            means = np.zeros(im_size)
            example, image_count, means = process_example(
                it_f=volume,
                it_l=label,
                means=means,
                im_size=im_size,
                label_size=label_size,
                targets=targets,
                repeat_image=repeat_image,
                preprocess=preprocess,
                normalize_im=normalize_im,
                store_z=store_z,
                tfrecord_writer=tfrecord_writer,
                metas=train_metas,
                augment=augment,
                augmentations=it_augmentations,
                input_shape=INPUT_SHAPE,
                label_shape=LABEL_SHAPE,
                image_count=image_count)

            if store_z and not augment:
                means = np.asarray(means).reshape(len(means), -1)
                np.savez(
                    '%s_means' % (it_ds_name),
                    volume={
                        'mean': means.mean(),
                        'std': means.std()
                    })
            else:
                np.save(
                    '%s_means' % (it_ds_name),
                    means / float(image_count))
    print('Finished %s with %s volumes' % (
        it_ds_name, image_count))

    # Test set
    it_ds_name = '%s_%s.tfrecords' % ('test', f_string)
    with tf.python_io.TFRecordWriter(it_ds_name) as tfrecord_writer:
        image_count = 0
        for f_idx, f in tqdm(
                enumerate(test_files),
                desc='Multidataset test',
                total=len(test_files)):
            dataset = np.load(f)
            volume, label = dataset['volume'], dataset['label']
            if affinity > 1:
                label = derive_affinities(
                    affinity=affinity,
                    label_volume=label)
            it_augmentations = check_augmentations(
                ND_TEST_AUGMENTATIONS)
            it_augmentations += add_test_augmentations
            means = np.zeros(im_size)
            example, image_count, means = process_example(
                it_f=volume,
                it_l=label,
                means=means,
                im_size=im_size,
                label_size=label_size,
                targets=targets,
                repeat_image=repeat_image,
                preprocess=preprocess,
                normalize_im=normalize_im,
                store_z=store_z,
                tfrecord_writer=tfrecord_writer,
                metas=test_metas,
                augment=augment,
                augmentations=it_augmentations,
                input_shape=INPUT_SHAPE,
                label_shape=LABEL_SHAPE,
                image_count=image_count)

            if store_z and not augment:
                means = np.asarray(means).reshape(len(means), -1)
                np.savez(
                    '%s_means' % (it_ds_name),
                    volume={
                        'mean': means.mean(),
                        'std': means.std()
                    })
            else:
                np.save(
                    '%s_means' % (it_ds_name),
                    means / float(image_count))
            print('Finished %s with %s volumes' % (
                it_ds_name, image_count))


def data_to_tfrecords(
        files,
        ds_name,
        im_size,
        label_size,
        preprocess,
        store_z=False,
        normalize_im=False,
        it_ds_name=None,
        augment=False,
        affinity=1,
        held_out=1,
        metas=None,
        no_npz=False,
        repeat_image=False):
    """Convert dataset to tfrecords."""
    if isinstance(files, list):
        # Create multidataset tfrecords
        create_multidataset(
            files=files,
            targets=targets,
            ds_name=ds_name,
            im_size=im_size,
            label_size=label_size,
            preprocess=preprocess,
            store_z=store_z,
            normalize_im=normalize_im,
            it_ds_name=it_ds_name,
            augment=augment,
            affinity=affinity,
            held_out=held_out,
            metas=metas,
            repeat_image=repeat_image)
    else:
        print(('Building dataset: {}'.format(ds_name)))
        print(('With key names: {}, {}'.format(*list(files.keys()))))
        for idx, (fk, fv) in enumerate(files.items()):
            if fk == 'test' or fk == 'val':
                sel_augmentations = ND_TEST_AUGMENTATIONS
            elif fk == 'train':
                sel_augmentations = ND_TRAIN_AUGMENTATIONS
            else:
                raise NotImplementedError(fk)
            it_ds_name = '%s_%s.tfrecords' % (ds_name, fk)
            if store_z:
                means = []
            else:
                means = np.zeros((im_size))
            with tf.python_io.TFRecordWriter(it_ds_name) as tfrecord_writer:
                image_count = 0
                for it_f in tqdm(
                        fv,
                        total=len(fv),
                        disable=augment,
                        desc='Building %s' % fk):
                    it_augmentations = check_augmentations(
                        sel_augmentations)
                    it_f = np.load(it_f)
                    vol = it_f['vol']
                    label = it_f['label'].reshape(label_size)
                    check = label.sum() > 0
                    if check:
                        example, image_count, means = process_example(
                            it_f=vol,
                            it_l=label,
                            means=means,
                            im_size=im_size,
                            label_size=label_size,
                            targets=TARGETS,
                            repeat_image=repeat_image,
                            preprocess=preprocess,
                            normalize_im=normalize_im,
                            store_z=store_z,
                            tfrecord_writer=tfrecord_writer,
                            metas=metas,
                            augment=augment,
                            augmentations=it_augmentations,
                            input_shape=im_size,
                            label_shape=label_size,
                            image_count=image_count)
                del it_f.f
                it_f.close()
                if not no_npz:
                    if store_z and not augment:
                        means = np.asarray(means).reshape(len(means), -1)
                        np.savez(
                            '%s_%s_means' % (ds_name, fk),
                            volume={
                                'mean': means.mean(),
                                'std': means.std()
                            })
                    else:
                        np.save(
                            '%s_%s_means' % (
                                ds_name, fk), means / float(image_count))
                print('Finished %s with %s volumes (dropped %s)' % (
                    it_ds_name, image_count, len(fv) - image_count))

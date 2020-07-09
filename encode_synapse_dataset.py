#!/usr/bin/env python
import os
import numpy as np
from config import Config
from utils import hybrid_utils as py_utils
from argparse import ArgumentParser
from ops.data_to_tfrecords import data_to_tfrecords
from tqdm import tqdm
from glob import glob


def pad_zeros(x, total):
    """Pad x with zeros to total digits."""
    num_pad = total - len(x)
    for idx in range(num_pad):
        x = '0' + x
    return x


def create_shards(
        it_shards,
        shard_dir,
        key,
        files,
        labels,
        targets,
        im_size,
        label_size,
        preprocess,
        store_z,
        normalize_im,
        affinity,
        augment=True,
        metas=None):
    """Build shards in a loop."""
    all_files = files[key]
    all_labels = labels[key]
    if key == 'test':
        total_data = len(all_files) / it_shards
        mask = np.arange(
            it_shards).reshape(1, -1).repeat(
                total_data).reshape(-1)
        all_files = all_files[:len(mask)]
        all_labels = all_labels[:len(mask)]
    total_shards = pad_zeros(str(it_shards), 5)
    for idx in tqdm(
            range(it_shards), total=it_shards, desc='Building %s' % key):
        shard_label = pad_zeros(str(idx), 5)
        shard_name = os.path.join(
            shard_dir,
            '%s-%s-of-%s.tfrecords' % (key, shard_label, total_shards))
        if key == 'test':
            it_mask = mask == idx
            it_files = {key: all_files[it_mask]}
            it_labels = {key: all_labels[it_mask]}
        else:
            it_files = {key: all_files}  # [it_mask]}
            it_labels = {key: all_labels}  # [it_mask]}
        data_to_tfrecords(
            files=it_files,
            labels=it_labels,
            targets=targets,
            ds_name=shard_name,
            im_size=im_size,
            label_size=label_size,
            preprocess=preprocess,
            store_z=store_z,
            it_ds_name=shard_name,
            augment=augment,
            normalize_im=normalize_im,
            affinity=affinity,
            no_npz=True,
            metas=metas)


def encode_dataset(
        output_name,
        train_shards=0,
        test_shards=0,
        force_test=False,
        augment=0,
        held_out=1,
        preproc_list=[],
        affinity=1,
        store_z=False,
        normalize_im=False):
    """Encode the synapses as tfrecords."""
    config = Config()
    files = np.array(glob(os.path.join(config.synapse_vols, '*.npz')))
    files = files[np.random.permutation(len(files))]
    data = np.load(files[0])
    im, lab = data['vol'], data['label']
    im_size = im.shape
    label_size = (1,) + lab.shape
    assert len(label_size) == 5
    assert len(im_size) == 5
    print('Label size: {}, Image size: {}'.format(label_size, im_size))
    del data.f
    data.close()

    # Split into CV folds
    sts = np.array([np.load(f)['row'][-1] for f in files])
    uni, ids, counts = np.unique(sts, return_inverse=True, return_counts=True)
    val_counts = np.array([np.round(co * .9) for co in counts]).astype(int)
    train_idx = np.zeros((len(sts)), dtype=np.bool)
    for la, th in enumerate(val_counts):
        it_train = np.where(ids == la)[0][:th]
        train_idx[it_train] = True
    val_idx = train_idx == False  # noqa
    datasets = {
        'train': files[train_idx],
        'val': files[val_idx],
    }
    if not train_shards:
        ds_name = os.path.join(config.tf_records, output_name)
        data_to_tfrecords(
            files=datasets,
            ds_name=ds_name,
            im_size=im_size,
            label_size=label_size,
            preprocess=preproc_list,
            store_z=store_z,
            affinity=affinity,
            augment=augment,
            normalize_im=normalize_im)
    else:
        assert test_shards > 0, 'Choose the number of test shards.'
        shard_dir = os.path.join(
            config.tf_records,
            data_proc.output_name)
        py_utils.make_dir(shard_dir)
        if not force_test:
            create_shards(
                it_shards=train_shards,
                shard_dir=shard_dir,
                key='train',
                files=files,
                labels=labels,
                im_size=im_size,
                label_size=label_size,
                preprocess=preproc_list,
                store_z=store_z,
                augment=augment,
                affinity=affinity,
                metas=metas,
                normalize_im=normalize_im)
        create_shards(
            it_shards=test_shards,
            shard_dir=shard_dir,
            key='test',
            files=files,
            labels=labels,
            im_size=im_size,
            label_size=label_size,
            preprocess=preproc_list,
            store_z=store_z,
            affinity=affinity,
            metas=metas,
            augment=augment,
            normalize_im=normalize_im)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--output_name',
        type=str,
        default='synapses_v7_smaller_label',
        dest='output_name',
        help='TF record name.')
    parser.add_argument(
        '--train_shards',
        type=int,
        default=0,
        dest='train_shards',
        help='Number of train shards for the dataset.')
    parser.add_argument(
        '--test_shards',
        type=int,
        default=128,
        dest='test_shards',
        help='Number of test shards for the dataset.')
    parser.add_argument(
        '--augment',
        type=int,
        default=0,
        dest='augment',
        help='Number of random augmentations.')
    parser.add_argument(
        '--held_out',
        type=int,
        default=1,
        dest='held_out',
        help='Number of datasets to hold out.')
    parser.add_argument(
        '--force_test',
        dest='force_test',
        action='store_true',
        help='Force creation of test dataset.')
    args = parser.parse_args()
    encode_dataset(**vars(args))

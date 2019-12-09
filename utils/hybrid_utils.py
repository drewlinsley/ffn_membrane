#!/usr/bin/env python
# __doc__ = """
import os
import numpy as np


def _bump_logit(z, y, x, t=0.5):
    return -(x * (1 - x))**(-t) - (y * (1 - y))**(-t) - (z * (1 - z))**(-t)


def _bump_logit_map(dim):
    x = range(dim[-1])
    y = range(dim[-2])
    z = range(dim[-3])
    zv, yv, xv = np.meshgrid(z, y, x, indexing='ij')
    xv = (xv + 1.0) / (dim[-1] + 1.0)
    yv = (yv + 1.0) / (dim[-2] + 1.0)
    zv = (zv + 1.0) / (dim[-3] + 1.0)
    return _bump_logit(zv, yv, xv)


def _bump_map(self, dim, max_logit, bump_map=None):
    if bump_map is None:
        return np.exp(_bump_logit_map(dim) - max_logit)
    else:
        return np.exp(bump_map - max_logit)


def recursive_make_dir(path, s=3):
    """Recursively build output paths."""
    split_path = path.split(os.path.sep)
    for idx, p in enumerate(split_path):
        if idx > s:
            d = '/'.join(split_path[:idx])
            if not os.path.exists(d):
                os.makedirs(d)
                # print('Created: %s' % d)
            else:
                # print('Reusing: %s' % d)
                pass


def pad_zeros(x, total):
    """Pad x with zeros to total digits."""
    if not isinstance(x, basestring):
        x = str(x)
    total = total - len(x)
    for idx in range(total):
        x = '0' + x
    return x


def make_dir(d):
    """Make directory d if it does not exist."""
    if not os.path.exists(d):
        os.makedirs(d)


def rdirs(coors, path, its=3):
    """Recursively make paths."""
    paths = path.split('/')
    for idx in reversed(range(its)):
        if idx == 2:
            it_path = '/'.join(paths[:-(idx + 1)]) % (pad_zeros(coors[0], 4))
        elif idx == 1:
            it_path = '/'.join(paths[:-(idx + 1)]) % (
                pad_zeros(coors[0], 4), pad_zeros(coors[1], 4))
        elif idx == 0:
            it_path = '/'.join(paths[:-(idx + 1)]) % (
                pad_zeros(coors[0], 4),
                pad_zeros(coors[1], 4),
                pad_zeros(coors[2], 4))
        make_dir(it_path)
        print 'Made: %s' % it_path

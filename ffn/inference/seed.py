# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Policies for choosing starting points for FFNs.

Seed policies are iterable objects yielding (z, y, x) tuples identifying
points at which the FFN will attempt to create a segment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import weakref

import numpy as np
from scipy import ndimage
import skimage
import skimage.feature

from . import storage


class BaseSeedPolicy(object):
  """Base class for seed policies."""

  def __init__(self, canvas, **kwargs):
    """Initializes the policy.

    Args:
      canvas: inference Canvas object; simple policies use this to access
          basic geometry information such as the shape of the subvolume;
          more complex policies can access the raw image data, etc.
      **kwargs: other keyword arguments
    """
    previous_origins = kwargs.get('previous_origins', None)
    previous_seg = kwargs.get('previous_seg', None)
    self.previous_origins = previous_origins
    self.previous_seg = previous_seg
    del kwargs
    # TODO(mjanusz): Remove circular reference between Canvas and seed policies.
    self.canvas = weakref.proxy(canvas)
    self.coords = None
    self.idx = 0

  def _init_coords(self):
    raise NotImplementedError()

  def __iter__(self):
    return self

  def __next__(self):
    """Returns the next seed point as (z, y, x).

    Does initial filtering of seed points to exclude locations that are
    too close to the image border.

    Returns:
      (z, y, x) tuples.

    Raises:
      StopIteration when the seeds are exhausted.
    """
    if self.coords is None:
      self._init_coords()

    while self.idx < self.coords.shape[0]:
      curr = self.coords[self.idx, :]
      self.idx += 1

      # TODO(mjanusz): Get rid of this.
      # Do early filtering of clearly invalid locations (too close to image
      # borders) as late filtering might be expensive.
      # if (np.all(curr - self.canvas.margin >= 0) and
      #     np.all(curr + self.canvas.margin <= self.canvas.shape)):
      return self.idx, tuple(curr)  # z, y, x

    raise StopIteration()

  def next(self):
    return self.__next__()

  def get_state(self):
    return self.coords, self.idx

  def set_state(self, state):
    self.coords, self.idx = state


class PolicyPeaks(BaseSeedPolicy):
  """Attempts to find points away from edges in the image.

  Runs a 3d Sobel filter to detect edges in the raw data, followed
  by a distance transform and peak finding to identify seed points.
  """

  def _init_coords(self, membrane_bias=False):
    logging.info('peaks: starting')

    # Edge detection.
    im = self.canvas.image.astype(np.float32)
    if membrane_bias:
        im = im[..., 0] * ((im[..., 1] / 2) + 0.5)
    else:
        im = im[..., 0]
    edges = ndimage.generic_gradient_magnitude(
        im,
        ndimage.sobel)

    # Adaptive thresholding.
    sigma = 49.0 / 6.0
    thresh_image = np.zeros(edges.shape, dtype=np.float32)
    ndimage.gaussian_filter(edges, sigma, output=thresh_image, mode='reflect')
    filt_edges = edges > thresh_image

    del edges, thresh_image

    # This prevents a border effect where the large amount of masked area
    # screws up the distance transform below.
    if (self.canvas.restrictor is not None and
        self.canvas.restrictor.mask is not None):
      filt_edges[self.canvas.restrictor.mask] = 1

    logging.info('peaks: filtering done')
    dt = ndimage.distance_transform_edt(1 - filt_edges).astype(np.float32)
    logging.info('peaks: edt done')

    # Use a specifc seed for the noise so that results are reproducible
    # regardless of what happens before the policy is called.
    state = np.random.get_state()
    np.random.seed(42)
    idxs = skimage.feature.peak_local_max(
        dt + np.random.random(dt.shape) * 1e-4,
        indices=True, min_distance=3, threshold_abs=0, threshold_rel=0)
    np.random.set_state(state)

    # After skimage upgrade to 0.13.0 peak_local_max returns peaks in
    # descending order, versus ascending order previously.  Sort ascending to
    # maintain historic behavior.
    idxs = np.array(sorted((z, y, x) for z, y, x in idxs))

    logging.info('peaks: found %d local maxima', idxs.shape[0])
    self.coords = idxs


class PolicyMembrane(BaseSeedPolicy):
  """Attempts to find points away from edges in the image.

  Runs a 3d Sobel filter to detect edges in the raw data, followed
  by a distance transform and peak finding to identify seed points.
  """

  def _init_coords(self):
    logging.info('peaks: starting')

    # Edge detection.
    edges = (
        self.canvas.image.astype(np.float32)[..., 1] > 0.5).astype(
            np.float32)

    # Distance transform
    logging.info('peaks: filtering done')
    dt = ndimage.distance_transform_edt(edges).astype(np.float32)
    logging.info('peaks: edt done')

    # Only keep seeds where there isn't a segmentation already
    # if hasattr(self.canvas, 'shifts'):
    #     # mask = (self.canvas.segmentation == 0).astype(np.float32)
    #     # dt *= mask
    #     shifts = np.array(self.canvas.shifts)
    #     nz = np.where(shifts != 0)[0]
    #     if shifts[nz] > 0:
    #         shifts[nz] - 8
    #     else:
    #         shifts[nz] + 8
    #     if shifts[0] > 0:
    #         dt[:-shifts[0]] = 0
    #     elif shifts[0] < 0:
    #         dt[:shifts[0]] = 0
    #     if shifts[1] > 0:
    #         dt[:, :-shifts[1]] = 0
    #     elif shifts[1] < 0:
    #         dt[:, :shifts[1]] = 0
    #     if shifts[2] > 0:
    #         dt[..., :-shifts[2]] = 0
    #     elif shifts[2] < 0:
    #         dt[..., :shifts[2]] = 0

    # Use a specifc seed for the noise so that results are reproducible
    # regardless of what happens before the policy is called.
    state = np.random.get_state()
    np.random.seed(42)
    idxs = skimage.feature.peak_local_max(
        dt + np.random.random(dt.shape) * 1e-4,
        indices=True, min_distance=3, threshold_abs=0, threshold_rel=0)
    np.random.set_state(state)

    # Sort by dt value
    idx_vals = [dt[ix[0], ix[1], ix[2]] for ix in idxs]
    sorted_idxs = np.argsort(idx_vals)[::-1]
    idxs = idxs[sorted_idxs]
    if hasattr(self.canvas, 'shifts') and self.canvas.shifts is not None:
        # Make sure there's overlap between new/old vols
        # Throw out seeds that were in the previous volume
        sz, sy, sx = self.canvas.shifts
        pre_idxs = len(idxs)
        z, y, x = self.canvas.segmentation.shape
        if sz > 0:
            idxs = idxs[idxs[:, 0] >= z - sz]
        elif sz < 0:
            idxs = idxs[idxs[:, 0] < z + sz]
        if sy > 0:
            idxs = idxs[idxs[:, 1] >= y - sy]
        elif sy < 0:
            idxs = idxs[idxs[:, 1] < y + sy]
        if sx > 0:
            idxs = idxs[idxs[:, 2] >= x - sx]
        elif sx < 0:
            idxs = idxs[idxs[:, 2] < x + sx]
        print(
          'Trimmed %s/%s seeds. (%s to process now).' % (
          pre_idxs - len(idxs), pre_idxs, len(idxs)))
        # h, w, d = self.canvas.segmentation.shape
        # if shifts[0] > 0:
        #     idxs = idxs[idxs[:, 0] >= h - shifts[0]]
        # elif shifts[0] < 0:
        #     idxs = idxs[idxs[:, 0] < shifts[0]]
        # if shifts[1] > 0:
        #     idxs = idxs[idxs[:, 1] >= w - shifts[1]]
        # elif shifts[1] < 0:
        #     idxs = idxs[idxs[:, 1] < shifts[1]]
        # if shifts[2] > 0:
        #     idxs = idxs[idxs[:, 2] >= d - shifts[2]]
        # elif shifts[2] < 0:
        #     idxs = idxs[idxs[:, 2] < shifts[2]]

    if self.previous_seg is not None:
      # Cycle through each of the segments > 0 in self.previous_seg. Choose the point with > dt.
      # Add this to the top of the list along with reserved ID.
      # 2. Cycle through each of the segs in self.previous_seg and get best idx
      unique_prev = np.unique(self.previous_seg).astype(int)
      unique_prev = unique_prev[unique_prev > 0]
      max_potential_idx, max_potentials = [], []
      margins = np.array(self.canvas.margin)  # // 2
      dt_mask = np.zeros_like(dt)
      dt_mask[
          margins[0]:-margins[0],
          margins[1]:-margins[1],
          margins[2]:-margins[2]] = 1.
      dt *= dt_mask
      for uidx in unique_prev:
        mask = self.previous_seg == uidx
        potential_idxs = dt * mask
        max_potential = np.max(potential_idxs)
        max_potential_id = np.unravel_index(
          potential_idxs.ravel().argmax(),
          dt.shape)
        max_potentials += [max_potential]
        max_potential_idx += [[max_potential_id]]
      sort_idx = np.argsort(max_potentials)[::-1]
      max_potential_idx = np.array(max_potential_idx)[sort_idx]
      uidxs = np.array(unique_prev)[sort_idx]

    # After skimage upgrade to 0.13.0 peak_local_max returns peaks in
    # descending order, versus ascending order previously.  Sort ascending to
    # maintain historic behavior.
    # idxs = np.array(sorted((z, y, x) for z, y, x in idxs))
    reserved_ids = [None] * len(idxs)
    if self.previous_seg is not None:
        idxs = np.concatenate((max_potential_idx.squeeze(1), idxs), 0)
        reserved_ids = uidxs.tolist() + reserved_ids
    nonzero_idxs = np.sum(idxs, 1) > 0
    idxs = idxs[nonzero_idxs]  # Throw out any weird 0,0,0 idxs
    reserved_ids = np.asarray(reserved_ids)[nonzero_idxs]
    logging.info('peaks: found %d local maxima', idxs.shape[0])
    self.coords = idxs
    self.reserved_ids = reserved_ids


class ShufflePolicyPeaks(BaseSeedPolicy):
  """Attempts to find points away from edges in the image.

  Runs a 3d Sobel filter to detect edges in the raw data, followed
  by a distance transform and peak finding to identify seed points.
  """

  def _init_coords(self, membrane_bias=False):
    logging.info('peaks: starting')
    # Edge detection.
    im = self.canvas.image.astype(np.float32)
    if membrane_bias:
        im = im[..., 0] * ((im[..., 1] / 2) + 0.5)
    else:
        im = im[..., 0]
    edges = ndimage.generic_gradient_magnitude(
        im,
        ndimage.sobel)

    # Adaptive thresholding.
    sigma = 49.0 / 6.0
    thresh_image = np.zeros(edges.shape, dtype=np.float32)
    ndimage.gaussian_filter(edges, sigma, output=thresh_image, mode='reflect')
    filt_edges = edges > thresh_image

    del edges, thresh_image

    # This prevents a border effect where the large amount of masked area
    # screws up the distance transform below.
    if (self.canvas.restrictor is not None and
        self.canvas.restrictor.mask is not None):
      filt_edges[self.canvas.restrictor.mask] = 1

    logging.info('peaks: filtering done')
    dt = ndimage.distance_transform_edt(1 - filt_edges).astype(np.float32)
    logging.info('peaks: edt done')

    # Use a specifc seed for the noise so that results are reproducible
    # regardless of what happens before the policy is called.
    state = np.random.get_state()
    np.random.seed(42)
    idxs = skimage.feature.peak_local_max(
        dt + np.random.random(dt.shape) * 1e-4,
        indices=True, min_distance=3, threshold_abs=0, threshold_rel=0)
    np.random.set_state(state)

    # After skimage upgrade to 0.13.0 peak_local_max returns peaks in
    # descending order, versus ascending order previously.  Sort ascending to
    # maintain historic behavior.
    idxs = np.array(sorted((z, y, x) for z, y, x in idxs))
    idxs = idxs[np.random.permutation(len(idxs))]  # Shuffle for ensembling

    logging.info('peaks: found %d local maxima', idxs.shape[0])
    self.coords = idxs


class PolicyPeaks2d(BaseSeedPolicy):
  """Attempts to find points away from edges at each 2d slice of image.

  Runs a 2d Sobel filter to detect edges in each 2d slice of
  raw data (specified by z index), followed by 2d distance transform
  and peak finding to identify seed points.
  """

  _SORT_CMP = dict(
      ascending=None,
      descending=lambda x, y: -cmp(x, y),
  )

  def __init__(self, canvas, min_distance=7, threshold_abs=2.5,
               sort_cmp='ascending', **kwargs):
    """Initialize settings.

    Args:
      canvas: inference Canvas object.
      min_distance: forwarded to peak_local_max.
      threshold_abs: forwarded to peak_local_max.
      sort_cmp: the cmp function to use for sorting seed coordinates.
      **kwargs: forwarded to base.

    For compatibility with original version, min_distance=3, threshold_abs=0,
    sort=False.
    """
    super(PolicyPeaks2d, self).__init__(canvas, **kwargs)
    self.min_distance = min_distance
    self.threshold_abs = threshold_abs
    self.sort_cmp = self._SORT_CMP[sort_cmp]

  def _init_coords(self):
    logging.info('2d peaks: starting')

    # Loop over 2d slices.
    for z in range(self.canvas.image.shape[0]):
      image_2d = (self.canvas.image[z, :, :]).astype(np.float32)

      # Edge detection.
      edges = ndimage.generic_gradient_magnitude(
          image_2d, ndimage.sobel)

      # Adaptive thresholding.
      sigma = 49.0 / 6.0
      thresh_image = np.zeros(edges.shape, dtype=np.float32)
      ndimage.gaussian_filter(edges, sigma, output=thresh_image, mode='reflect')
      filt_edges = edges > thresh_image

      del edges, thresh_image

      # Prevent border effect
      if (self.canvas.restrictor is not None and
          self.canvas.restrictor.mask is not None):
        filt_edges[self.canvas.restrictor.mask[z, :, :]] = 1

      # Distance transform
      dt = ndimage.distance_transform_edt(1 - filt_edges).astype(np.float32)

      # Use a specifc seed for the noise so that results are reproducible
      # regardless of what happens before the policy is called.
      state = np.random.get_state()
      np.random.seed(42)
      idxs = skimage.feature.peak_local_max(
          dt + np.random.random(dt.shape) * 1e-4,
          indices=True, min_distance=3, threshold_abs=0, threshold_rel=0)
      zs = np.full((idxs.shape[0], 1), z, dtype=np.int64)
      idxs = np.concatenate((zs, idxs), axis=1)
      np.random.set_state(state)

      # Update self.coords with indices found at this z index
      logging.info('2d peaks: found %d local maxima at z index %d',
                   idxs.shape[0], z)
      self.coords = np.concatenate((self.coords, idxs)) if z != 0 else idxs

    self.coords = np.array(
        sorted([(z, y, x) for z, y, x in self.coords], cmp=self.sort_cmp))

    logging.info('2d peaks: found %d total local maxima', self.coords.shape[0])


class PolicyMax(BaseSeedPolicy):
  """All points in the image, in descending order of intensity."""

  def _init_coords(self):
    idxs = np.mgrid[[slice(0, x) for x in self.canvas.image.shape]]
    sort_idx = np.argsort(self.canvas.image.flat)[::-1]
    self.coords = np.array(zip(*[idx.flat[sort_idx] for idx in idxs]))


class PolicyGrid3d(BaseSeedPolicy):
  """Points distributed on a uniform 3d grid."""

  def __init__(self, canvas, step=16, offsets=(0, 8, 4, 12, 2, 10, 14),
               **kwargs):
    super(PolicyGrid3d, self).__init__(canvas, **kwargs)
    self.step = step
    self.offsets = offsets

  def _init_coords(self):
    self.coords = []
    for offset in self.offsets:
      for z in range(offset, self.canvas.image.shape[0], self.step):
        for y in range(offset, self.canvas.image.shape[1], self.step):
          for x in range(offset, self.canvas.image.shape[2], self.step):
            self.coords.append((z, y, x))
    self.coords = np.array(self.coords)


class PolicyGrid2d(BaseSeedPolicy):
  """Points distributed on a uniform 2d grid."""

  def __init__(self, canvas, step=16, offsets=(0, 8, 4, 12, 2, 6, 10, 14),
               **kwargs):
    super(PolicyGrid2d, self).__init__(canvas, **kwargs)
    self.step = step
    self.offsets = offsets

  def _init_coords(self):
    self.coords = []
    for offset in self.offsets:
      for z in range(self.canvas.image.shape[0]):
        for y in range(offset, self.canvas.image.shape[1], self.step):
          for x in range(offset, self.canvas.image.shape[2], self.step):
            self.coords.append((z, y, x))
    self.coords = np.array(self.coords)


class PolicyInvertOrigins(BaseSeedPolicy):

  def __init__(self, canvas, corner=None, segmentation_dir=None,
               **kwargs):
    super(PolicyInvertOrigins, self).__init__(canvas, **kwargs)
    self.corner = corner
    self.segmentation_dir = segmentation_dir

  def _init_coords(self):
    origins_to_invert = storage.load_origins(self.segmentation_dir,
                                             self.corner)
    points = origins_to_invert.items()
    points.sort(reverse=True)
    self.coords = np.array([origin_info.start_zyx for _, origin_info
                            in points])

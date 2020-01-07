import os
import numpy as np
from ffn.inference import segmentation


def split_segmentation_by_intersection(a, b, min_size):
  """Computes the intersection of two segmentations.
  Intersects two spatially overlapping segmentations and assigns a new ID to
  every unique (id1, id2) pair of overlapping voxels. If 'id2' is the largest
  object overlapping 'id1', their intersection retains the 'id1' label. If the
  fragment created by intersection is smaller than 'min_size', it gets removed
  from the segmentation (assigned an id of 0 in the output).
  `a` is modified in place, `b` is not changed.
  Note that (id1, 0) is considered a valid pair and will be mapped to a non-zero
  ID as long as the size of the overlapping region is >= min_size, but (0, id2)
  will always be mapped to 0 in the output.
  Args:
    a: First segmentation.
    b: Second segmentation.
    min_size: Minimum size intersection segment to keep (not map to 0).
  Raises:
    TypeError: if a or b don't have a dtype of uint64
    ValueError: if a.shape != b.shape, or if `a` or `b` contain more than
                2**32-1 unique labels.
  """
  if a.shape != b.shape:
    raise ValueError
  a = a.ravel()
  output_array = a

  b = b.ravel()

  def remap_input(x):
    """Remaps `x` if needed to fit within a 32-bit ID space.
    Args:
      x: uint64 numpy array.
    Returns:
      `remapped, max_id, orig_values_map`, where:
        `remapped` contains the remapped version of `x` containing only
        values < 2**32.
        `max_id = x.max()`.
        `orig_values_map` is None if `remapped == x`, or otherwise an array such
        that `x = orig_values_map[remapped]`.
    Raises:
      TypeError: if `x` does not have uint64 dtype
      ValueError: if `x.max() > 2**32-1`.
    """
    if x.dtype != np.uint64:
      raise TypeError
    max_uint32 = 2**32 - 1
    max_id = x.max()
    orig_values_map = None
    if max_id > max_uint32:
      orig_values_map, x = np.unique(x, return_inverse=True)
      if len(orig_values_map) > max_uint32:
        raise ValueError('More than 2**32-1 unique labels not supported')
      x = np.cast[np.uint64](x)
      if orig_values_map[0] != 0:
        orig_values_map = np.concatenate(
            [np.array([0], dtype=np.uint64), orig_values_map])
        x[...] += 1
    return x, max_id, orig_values_map

  remapped_a, max_id, a_reverse_map = remap_input(a)
  remapped_b, _, _ = remap_input(b)

  intersection_segment_ids = np.bitwise_or(remapped_a, remapped_b << 32)

  unique_joint_labels, remapped_joint_labels, joint_counts = np.unique(
      intersection_segment_ids, return_inverse=True, return_counts=True)

  unique_joint_labels_a = np.bitwise_and(unique_joint_labels, 0xFFFFFFFF)
  unique_joint_labels_b = unique_joint_labels >> 32

  # Maps each segment id `id_a` in `remapped_a` to `(id_b, joint_count)` where
  # `id_b` is the segment id in `remapped_b` with maximum overlap, and
  # `joint_count` is the number of voxels of overlap.
  max_overlap_ids = dict()

  for label_a, label_b, count in zip(unique_joint_labels_a,
                                     unique_joint_labels_b, joint_counts):
    new_pair = (label_b, count)
    existing = max_overlap_ids.setdefault(label_a, new_pair)
    if existing[1] < count:
      max_overlap_ids[label_a] = new_pair

  # Relabel map to apply to remapped_joint_labels to obtain the output ids.
  new_labels = np.zeros(len(unique_joint_labels), np.uint64)
  for i, (label_a, label_b, count) in enumerate(zip(unique_joint_labels_a,
                                                    unique_joint_labels_b,
                                                    joint_counts)):
    if count < min_size or label_a == 0:
      new_label = 0
    elif label_b == max_overlap_ids[label_a][0]:
      if a_reverse_map is not None:
        new_label = a_reverse_map[label_a]
      else:
        new_label = label_a
    else:
      max_id += 1
      new_label = max_id
    new_labels[i] = new_label

  output_array[...] = new_labels[remapped_joint_labels]
  return output_array


# pre_path = '/users/dlinsley/ffn_v2/ding_segmentations/x0099/y0099/z0099/'
pre_path = '/users/dlinsley/ffn_v2/ding_segmentations/x0015/y0015/z0017'
file_path = '0/0/seg-0_0_0.npz'
min_size = 0  # 1024
split_min_size = 7

v1_path = os.path.join(pre_path, 'v1', file_path)
v2_path = os.path.join(pre_path, 'v2', file_path)
v1 = np.load(v1_path)['segmentation'].astype(np.uint64)
v2 = np.load(v2_path)['segmentation'].astype(np.uint64)

if min_size:
  v1 = segmentation.clean_up(seg=v1, min_size=min_size)
  v2 = segmentation.clean_up(seg=v2, min_size=min_size)

new_v1 = split_segmentation_by_intersection(v1, v2,split_min_size)
# new_v1 = segmentation.reduce_id_bits(new_v1)


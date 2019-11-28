"""Functions to be cythonized."""
from scipy import stats


def reassign_labels(unique_labels, segments, labeled_segments):
  """Reassign labels of overlapping segs to the modal seg.

  Requires:
  unique_labels = np.unique(labeled_segments)
  segments: original segmentations
  labeled_segments: preprocessed segmentations
  """
  for k in unique_labels:
    mask = labeled_segments == k
    cdef int m = mask
    seg_vec = segments * m
    seg_vec = seg_vec[seg_vec > 0]
    modal = stats.mode(seg_vec)[0][0]
    labeled_segments[mask] = modal
  return labeled_segments


